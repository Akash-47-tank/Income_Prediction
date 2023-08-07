
# CassandraDB/app_logging/logger.py

import logging


class Applogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fileHandler = logging.FileHandler("Training_Logs/GeneralLog.txt")
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.fileHandler.setFormatter(self.formatter)
        self.logger.addHandler(self.fileHandler)
        self.logger.setLevel(logging.INFO)

    def create_log_file(self, log_file_name):
        fileHandler = logging.FileHandler(log_file_name)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)
        self.logger.setLevel(logging.INFO)

    def log(self, file_object: object, log_message: object) -> object:
        """
        rtype: object
        """
        self.logger.info(log_message)
        file_object.write(log_message + "\n")
