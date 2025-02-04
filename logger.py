import logging


def get_logger(name):
    """
    Create and return a logger with the specified name.
    """
    logger = logging.getLogger(name)  # Set a custom name
    logger.setLevel(logging.INFO)  # Set the logging level

    file_handler = logging.FileHandler(f"logs/{name}.log")  # Unique log file per app
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger
