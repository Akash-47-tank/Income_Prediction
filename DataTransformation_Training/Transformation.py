import pandas as pd
import os
from CassandraDB.app_logging.logger import Applogger


class DataTransform:
    def __init__(self):
        self.good_data_path = "Training_Raw_files_validated/Good_Raw"
        self.logger = Applogger()

    def replace_missing_with_null(self):
        """
        Method Name: replaceMissingWithNull
        Description: This method replaces the missing values in columns with "Null" to prepare the data for insertion
                     to the database.
        """
        file_list = os.listdir(self.good_data_path)
        try:
            for file in file_list:
                csv = pd.read_csv(f"{self.good_data_path}/{file}")
                csv.fillna("Null", inplace=True)
                csv.to_csv(f"{self.good_data_path}/{file}", index=None, header=True)
                self.logger.log(f"{file}", "Missing values replaced with 'Null'")
        except Exception as e:
            self.logger.log("An error occurred while replacing missing values with 'Null'")
            self.logger.log(str(e))
            raise e
