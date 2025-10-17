# This code is based on Gilad's code in "Genome scale analysis of Escherichia coli with a comprehensive prokaryotic sequence-based biophysical model of translation initiation and elongation"

import functools
import logging
import subprocess
import parse
import math
from collections import deque, namedtuple

import pandas as pd
from nupack import *

import sys
sys.path.insert(1, '..')
from logging_utils import get_logger

WORKING_DIRECTORY = ""
LOG_FILE = "rRNA.log"
LOGGER_NAME = "feature_adding_logger"
POTENTIAL_TRIGGERS_FILE_PATH = "/Data/full_df.csv"
OUTPUT_FILE_PATH = "/Data/full_df_with_translation_initiation_features.csv"
rRNA = "acctcctta".upper().replace("T", "U") # These are the last 9 nt (3' end) of the 16S rRNA in E. coli (https://github.com/hsalis/Ribosome-Binding-Site-Calculator-v1.0/blob/master/RBS_Calculator.py)
dG_START_LOOKUP = {'AUG': -1.194, 'GUG': -0.0748, 'UUG': -0.0435, 'CUG': -0.03406}
CUTOFF = 35 # number of nt +- start codon considering for folding
STANDBY_LENGTH = 4

logger = get_logger(LOGGER_NAME, os.path.join(WORKING_DIRECTORY, LOG_FILE))

FoldResult = namedtuple("FoldResult", ["structure", "energy"])
DuplexTDataTuple = namedtuple("DuplexTDataTuple", ["energy", "mRnaPos", "rRnaPos", 'hairpins', "structure", "nstart"])

class DuplexTData(DuplexTDataTuple):
    def AlignedSpacing(self):
        # s = nstart − n1 − n2 , where n1 and n2 are the rRNA and mRNA nucleotide positions in the
        # farthest 3′ base pair in the 16S rRNA binding site.

        # The farthest 3' base pair in the 16S rRNA binding site is the last rRNA nt and since it binds in reverse order
        # then the mRNA, it is the first nt in the mRNA.
        # nstart and positions are 0 based, the calculation above is 1 based, which means we calculate
        # (nstart-1) - (n1-1) - (n2-1), so subtract 1 to fix
        if len(self.mRnaPos) > 0:
            return self.nstart - self.mRnaPos[0] - self.rRnaPos[-1] - 1
        else:
            return None

    def calcSpacingDg(self, s):
        sOpt = 5
        if s >= 5:
            c1 = 0.048
            c2 = 0.24
            return c1 * ((s - sOpt) ** 2) + c2 * (s - sOpt)
        else:
            c1 = 12.2
            c2 = 2.5
            return c1 / ((1 + math.exp(c2 * (s - sOpt + 2))) ** 3)

    def SpacingDg(self):
        return self.calcSpacingDg(self.AlignedSpacing())

class ViennaRna:
    def __init__(self, path):
        self.vienna_path = path
        self.fold_parse = parse.compile("{structure} ({energy:6.2f})")
        self.subopt_parse = parse.compile("{structure} {energy:6.2f}")

    @functools.lru_cache(maxsize=1000)
    def RnaFold(self, seq, useDangles=True):

        if useDangles:
            dangles = 2
        else:
            dangles = 0

        args = ["-d", dangles, "--noPS"]
        output, returncode = self._run_command("RNAfold", args, seq)

        if returncode != 0:
            raise Exception("Error calling RNAfold, return code %d" % (returncode))

        output = output.decode('utf-8')
        lines = output.splitlines()
        parseResult = self.fold_parse.parse(lines[1])

        return FoldResult(parseResult['structure'], parseResult['energy'])

    @functools.lru_cache(maxsize=10000)
    def RnaSubopt(self, sequences, deltaEnergy, useDangles):
        result = deque()
        seq_string = "&".join(sequences)

        if useDangles:
            dangles = 2
        else:
            dangles = 0

        args = ["-e", deltaEnergy, "-d", dangles]
        output, returncode = self._run_command("RNAsubopt", args, seq_string)

        if returncode != 0:
            raise Exception("Error calling RNAsubopt, return code %d" % (returncode))

        output = output.decode('utf-8')
        lines = output.splitlines()

        for line in lines[1:]:
            parseResult = self.subopt_parse.parse(line)
            result.append(FoldResult(parseResult['structure'], parseResult['energy']))

        return result

    @functools.lru_cache(maxsize=10000)
    def RnaEval(self, sequences, structure, useDangles):

        seqString = "&".join(sequences)

        if useDangles:
            dangles = 2
        else:
            dangles = 0

        args = ["-d", dangles]
        output, returncode = self._run_command("RNAeval", args, seqString + "\n" + structure)
        if returncode != 0:
            raise subprocess.SubprocessError("Error calling RNAeval, return code %d" % (returncode))

        output = output.decode('utf-8')
        lines = output.splitlines()
        parseResult = self.fold_parse.parse(lines[1])

        return FoldResult(parseResult['structure'], parseResult['energy'])

    def _run_command(self, cmd, args, inputString):
        args = [str(elem).strip() for elem in args]
        args = list(filter(None, args))

        prog = subprocess.Popen([os.path.join(self.vienna_path, cmd)] + args, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        output = prog.communicate(inputString.encode('utf-8'))[0]

        return output, prog.returncode


def calc_start_dG(start_codon: str) -> float:
    return dG_START_LOOKUP[start_codon.upper()]

def FoldStructureIterator(iterator, openChar, closeChar):
    # Call this to iterate over structure with folding and report all closing
    # characters that don't have an open one. This allows to filter internal
    # foldings. For example:
    # FoldStructureIterator(reversed( '.(((...)))((.'), ')', '(' ) will report the
    # last 2, but will skip the other 3 '(' characters
    # FoldStructureIterator( ')))()' , '(', ')' ) will report the first 3, but
    # will skip the 4th ')'
    openCount = 0
    startPos = None
    for i, e in iterator:
        if e == closeChar:
            if openCount > 0: openCount -= 1
            if openCount == 0:
                yield startPos, i
                startPos = None

        elif e == openChar:
            if openCount == 0: startPos = i
            openCount += 1

def DuplexTDataFromFoldResult(foldResult, nstart, offsetNt):
    positionActualStart = max(0, nstart - offsetNt)
    structure = foldResult.structure
    separator = structure.find('&')
    mRnaStructure = [r for r in FoldStructureIterator(reversed(list(enumerate(structure[:separator]))), ')', '(')]
    rRnaPos = [i for (s, i) in FoldStructureIterator(enumerate(structure[separator + 1:]), '(', ')') if s is None]
    hairpins = [(positionActualStart + i, positionActualStart + s) for (s, i) in mRnaStructure[::-1] if s is not None]
    mRnaPos = [positionActualStart + pos for (s, pos) in mRnaStructure[::-1] if s is None]

    return DuplexTData(foldResult.energy, mRnaPos, rRnaPos, hairpins, structure, nstart)

def get_result_from_duplex_subopt(subopt_result, start_pos):
    return (DuplexTDataFromFoldResult(fold_result, start_pos, CUTOFF) for fold_result in subopt_result)

def calc_standby_energy(useDangles, mRnaBeforeStart, duplexT, rRNA, viennaRna):
    dgStandby = 0
    mRnaFirstBindingSite = len(mRnaBeforeStart) - (duplexT.nstart - duplexT.mRnaPos[0])

    if mRnaFirstBindingSite > STANDBY_LENGTH:
        mRnaBeforeStandbySeq = mRnaBeforeStart[:mRnaFirstBindingSite - STANDBY_LENGTH]
        mRnaBeforeStandbyFold = viennaRna.RnaFold(mRnaBeforeStandbySeq, useDangles)
        standbyStructure = mRnaBeforeStandbyFold.structure + "." * STANDBY_LENGTH

        if standbyStructure != duplexT.structure[:len(standbyStructure)]:
            mRnaLastBindingSite = len(mRnaBeforeStart) - (duplexT.nstart - duplexT.mRnaPos[-1])
            newStructure = standbyStructure + duplexT.structure[mRnaFirstBindingSite:mRnaLastBindingSite + 1] + "." * (len(mRnaBeforeStart) - (mRnaLastBindingSite + 1)) + duplexT.structure[len(mRnaBeforeStart):]
            mRnaAfterStandbyFold = viennaRna.RnaEval((mRnaBeforeStart, rRNA), newStructure, useDangles)
            dgStandby = duplexT.energy - mRnaAfterStandbyFold.energy

    return dgStandby

def calc_init_rate(dG):
    # boltzmannFactor = 0.45 (Salis et al., 2009; Fig. 2.3).
    boltzmannFactor = 0.45
    # We choose K = 2500 so that physiologically possible translation initiation rates r
    # will vary between 0.1 and 100,000.
    K = 2500
    return K * math.exp(-boltzmannFactor * dG)

def get_translation_initiation_energy_features(mRNA: str, start_pos: int):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.info("calculating translation initiation features")

    viennaRna = ViennaRna("/bin")

    use_dangles = start_pos < CUTOFF

    start_codon = mRNA[start_pos:start_pos + 3]
    start_dG = calc_start_dG(start_codon)

    m_rna_before_start = mRNA[max(0, start_pos - CUTOFF):start_pos]
    m_rna_after_start = mRNA[start_pos:min(start_pos + CUTOFF, len(mRNA))]

    mRNA_dG = viennaRna.RnaFold(m_rna_before_start + m_rna_after_start, use_dangles)

    duplex_subopt = viennaRna.RnaSubopt((m_rna_before_start, rRNA), 3, use_dangles)
    duplex_subopt_list = [item for item in get_result_from_duplex_subopt(duplex_subopt, start_pos)
                          if len(item.mRnaPos) > 0]

    if len(duplex_subopt_list) == 0:
        return None, None, None, start_dG, None, mRNA_dG.energy, None
        # raise Exception("No suitable ribosome binding site found")

    # Find the suboptimal folding that minimizes its energy + spacing energy
    rRNA_mRNA_duplex_dG = min(duplex_subopt_list, key=lambda e: e.energy + e.SpacingDg())

    # Calculate the standby energy for the minimum
    standby_dG = calc_standby_energy(use_dangles, m_rna_before_start, rRNA_mRNA_duplex_dG, rRNA, viennaRna)

    total_energy = (rRNA_mRNA_duplex_dG.energy + start_dG + rRNA_mRNA_duplex_dG.SpacingDg() + standby_dG) - mRNA_dG.energy
    init_rate = calc_init_rate(total_energy)

    return (total_energy, init_rate, rRNA_mRNA_duplex_dG.energy, start_dG, rRNA_mRNA_duplex_dG.SpacingDg(),
            mRNA_dG.energy, standby_dG)


if __name__ == '__main__':
    triggers_df = pd.read_csv(POTENTIAL_TRIGGERS_FILE_PATH)
    triggers_df[[
        "translation_initiation_energy", "translation_initiation_rate", "rRNA_mRNA_duplex_dG",
        "start_dG", "spacing_dG", "mRNA_dG", "standby_dG"
    ]] = triggers_df.apply(
        lambda row: get_translation_initiation_energy_features(row["switch"], row["stem_top_end"] + 1),
        axis=1, result_type="expand"
    )
    triggers_df.to_csv(OUTPUT_FILE_PATH)
