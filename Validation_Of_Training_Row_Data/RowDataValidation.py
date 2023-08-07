import sqlite3
from datetime import datetime
import os
import re
import json
import shutil
import pandas as pd
from CassandraDB.app_logging.logger import Applogger


class RawDataValidator:
    def __init__(self, path):
        self.file_object = None
        self.batch_directory = path
        self.schema_path = 'Schema_Training_Format.json'
        self.logger = Applogger()

    def get_values_from_schema(self):
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)

            date_stamp_length = schema.get('LengthOfDateStampInFile')
            time_stamp_length = schema.get('LengthOfTimeStampInFile')
            column_names = schema.get('ColName')
            num_of_columns = schema.get('NumberColumns')

            if date_stamp_length is None or time_stamp_length is None or column_names is None or num_of_columns is None:
                raise KeyError("One or more keys are missing or have invalid values in the schema_training.json file.")

            file = open("Training_Logs/values fromSchemaValidationLog.txt", 'a+')
            message = f"LengthOfDateStampInFile: {date_stamp_length}\tLengthOfTimeStampInFile: {time_stamp_length}\tNumberColumns: {num_of_columns}\n"
            self.logger.log(file, message)
            file.close()

        except ValueError:
            file = open("Training_Logs/values fromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, "ValueError: Value not found inside schema_training.json")
            file.close()
            raise ValueError

        except KeyError as e:
            file = open("Training_Logs/values fromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, f"KeyError: {str(e)}")
            file.close()
            raise e

        except Exception as e:
            file = open("Training_Logs/values fromSchemaValidationLog.txt", 'a+')
            self.logger.log(file, str(e))
            file.close()
            raise e

        return date_stamp_length, time_stamp_length, column_names, num_of_columns

    def manual_regex_creation(self):
        regex = r'CensusIncomeData_\d{9}_\d{6}\.csv'
        return regex

    def create_directory_for_good_bad_raw_data(self):
        try:
            path = os.path.join("Training_Raw_files_validated", "Good_Raw")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join("Training_Raw_files_validated", "Bad_Raw")
            if not os.path.isdir(path):
                os.makedirs(path)

        except OSError as ex:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, f"Error while creating directories: {ex}")
            file.close()
            raise OSError

    def delete_existing_good_data_training_folder(self):
        try:
            path = 'Training_Raw_files_validated/Good_Raw'
            if os.path.isdir(path):
                shutil.rmtree(path)
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, "GoodRaw directory deleted successfully!!!")
                file.close()
        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, f"Error while deleting Good Raw Data folder: {s}")
            file.close()
            raise OSError

    def delete_existing_bad_data_training_folder(self):
        try:
            path = 'Training_Raw_files_validated/Bad_Raw'
            if os.path.isdir(path):
                shutil.rmtree(path)
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, "BadRaw directory deleted before starting validation!!!")
                file.close()
        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, f"Error while deleting Bad Raw Data folder: {s}")
            file.close()
            raise OSError

    def move_bad_files_to_archive_bad(self):
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            source = 'Training_Raw_files_validated/Bad_Raw'
            if os.path.isdir(source):
                dest = f"TrainingArchiveBadData/BadData_{date}_{time}"
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(os.path.join(source, f), dest)
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, "Bad files moved to archive")
                path = 'Training_Raw_files_validated/Bad_Raw'
                if os.path.isdir(path):
                    shutil.rmtree(path)
                self.logger.log(file, "Bad Raw Data Folder Deleted successfully!!")
                file.close()
        except Exception as e:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, f"Error while moving bad files to archive: {e}")
            file.close()
            raise e

    def validate_file_name_raw(self, regex, date_stamp_length, time_stamp_length):
        self.delete_existing_bad_data_training_folder()
        self.delete_existing_good_data_training_folder()
        self.create_directory_for_good_bad_raw_data()

        only_files = [f for f in os.listdir(self.batch_directory)]
        try:
            file = open("Training_Logs/nameValidationLog.txt", 'a+')
            for filename in only_files:
                if re.match(regex, filename):
                    split_at_dot = filename.split('.csv')[0]
                    split_at_dot = split_at_dot.split('_')

                    if len(split_at_dot[1]) == date_stamp_length and len(split_at_dot[2]) == time_stamp_length:
                        shutil.copy(os.path.join(self.batch_directory, filename), "Training_Raw_files_validated/Good_Raw")
                        self.logger.log(file, f"Valid File name!! File moved to GoodRaw Folder: {filename}")
                    else:
                        shutil.copy(os.path.join(self.batch_directory, filename), "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(file, f"Invalid File Name!! File moved to Bad Raw Folder: {filename}")
                else:
                    shutil.copy(os.path.join(self.batch_directory, filename), "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(file, f"Invalid File Name!! File moved to Bad Raw Folder: {filename}")

            file.close()

        except Exception as e:
            file = open("Training_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(file, f"Error occurred while validating FileName: {e}")
            file.close()
            raise e

    def validate_column_length(self, num_of_columns):
        try:
            file = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(file, "Column Length Validation Started!!")

            for filename in os.listdir('Training_Raw_files_validated/Good_Raw'):
                csv = pd.read_csv(os.path.join('Training_Raw_files_validated/Good_Raw', filename))
                if csv.shape[1] == num_of_columns:
                    pass
                else:
                    shutil.move(os.path.join('Training_Raw_files_validated/Good_Raw', filename),
                                "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(file, f"Invalid Column Length for the file!! File moved to Bad Raw Folder: {filename}")

            self.logger.log(file, "Column Length Validation Completed!!")

        except OSError:
            file = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(file, "Error Occurred while moving the file :: OSError")
            file.close()
            raise OSError

        except Exception as e:
            file = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(file, f"Error Occurred :: {e}")
            file.close()
            raise e

        file.close()

    def deletePredictionFile(self):
        """
        Method Name: deletePredictionFile
        Description: This method deletes the prediction output file if it exists.
        """

        try:
            prediction_file_path = "Prediction_Output_File/Predictions.csv"
            if os.path.exists(prediction_file_path):
                os.remove(prediction_file_path)
                self.logger.log(self.file_object, "Prediction output file deleted successfully!")
            else:
                self.logger.log(self.file_object, "Prediction output file not found. Nothing to delete.")
        except Exception as ex:
            self.logger.log(self.file_object, "Error occurred while deleting the prediction output file. Error:: %s" % ex)
            raise ex

    def validate_missing_values_in_whole_column(self):
        try:
            file = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(file, "Missing Values Validation Started!!")

            for filename in os.listdir('Training_Raw_files_validated/Good_Raw'):
                csv = pd.read_csv(os.path.join('Training_Raw_files_validated/Good_Raw', filename))
                count = 0
                for column in csv:
                    if (len(csv[column]) - csv[column].count()) == len(csv[column]):
                        count += 1
                        shutil.move(os.path.join('Training_Raw_files_validated/Good_Raw', filename),
                                    "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(file, f"Invalid Column for the file!! File moved to Bad Raw Folder: {filename}")
                        break

                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv(os.path.join('Training_Raw_files_validated/Good_Raw', filename), index=None, header=True)

        except OSError:
            file = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(file, "Error Occurred while moving the file :: OSError")
            file.close()
            raise OSError

        except Exception as e:
            file = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(file, f"Error Occurred :: {e}")
            file.close()
            raise e

        file.close()
