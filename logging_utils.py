import logging

FORMAT = '%(asctime)s %(filename)s:%(lineno)s %(levelname)s: %(message)s'


def init_logging(path):
    logging.basicConfig(filename=path, encoding='utf-8', level=logging.INFO, format=FORMAT)


def get_logger(name, path):
    FORMAT = '%(asctime)s %(filename)s:%(lineno)s %(levelname)s: %(message)s'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    format = logging.Formatter(FORMAT)
    file_handler.setFormatter(format)
    logger.addHandler(file_handler)
    return logger
