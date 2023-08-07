from datetime import datetime

from CassandraDB.app_logging import logger
from Validation_Of_Training_Row_Data.RowDataValidation import RawDataValidator
from TrainingData_Validation_Insertation_DBoperation.DataType_Validation_DBoperation import DBOperations
from DataTransformation_Training.Transformation import DataTransform
from CassandraDB.app_logging.logger import Applogger


class TrainValidation:
    def __init__(self, path):
        self.RawDataValidator = RawDataValidator(path)
        self.data_transformer = DataTransform()
        self.db_operator = DBOperations()
        self.log_writer = logger.Applogger()
        self.log_writer.create_log_file("Training_Logs/Training_Main_Log.txt")
        self.file_object = open("CassandraDB/Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer = logger.Applogger()

    def perform_validation(self):
        # Provide the path to your data folder here
        data_folder_path = "/Users/aakash/PycharmProjects/income_ineuron/CassandraDB/dataset"

        # Provide the path to your schema file here
        schema_file_path = "/Users/aakash/PycharmProjects/income_ineuron/CassandraDB/Schema_Training_Format.json"

        validator = TrainValidation(data_folder_path)
        db_operations = DBOperations()

        try:
            self.log_writer.log(self.file_object, 'Start of Validation on files for prediction !!')

            # Extracting values from prediction schema format
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, num_of_columns = self.RawDataValidator.get_values_from_schema()

            # creating the regex to validate filename
            regex = self.RawDataValidator.manual_regex_creation()

            # Validating filename of prediction files
            self.RawDataValidator.validate_file_name_raw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)

            # Validating column length in the file
            self.RawDataValidator.validate_column_length(num_of_columns)

            # Validating if any column has all values missing
            self.RawDataValidator.validate_missing_values_in_whole_column()
            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            # Transformation
            self.log_writer.log(self.file_object, "Starting Data Transformation!!")
            # Replacing blanks in the CSV file with "Null" values to insert in table
            self.data_transformer.replace_missing_with_null()
            self.log_writer.log(self.file_object, "Data Transformation Completed!!!")

            # DataBase Operation

            # Perform raw data validation, transformation, and database operations
            validator.perform_validation()

            # Read the schema file to get column names and data types
            schema_data = db_operations.read_schema_file(schema_file_path)
            column_names = schema_data["ColName"]


            self.log_writer.log(self.file_object,
                                "Creating Training_Database and tables on the basis of given schema!!!")
            # Assuming 'data_base_connection' method exists in DBOperations class
            session = self.db_operator.data_base_connection()

            # Insert CSV files into the table
            self.db_operator.insert_into_table('incomedb',"incomedb.adult2",column_names)
            self.log_writer.log(self.file_object, "Insertion in Table completed!!!")
            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")

            # Delete the good data folder after loading files in the table
            self.RawDataValidator.delete_existing_good_data_training_folder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")
            self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")

            # Move the bad files to the archive folder
            self.RawDataValidator.move_bad_files_to_archive_bad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object, "Validation Operation completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")

            # model training is going to happen with combined data which we have inserted into DataBase
            # Assuming 'selecting_data_from_table_into_csv' method exists in DBOperations class
            self.db_operator.selecting_data_from_table_into_csv()

            self.file_object.close()

        except Exception as e:
            # If any exception occurs, it will be raised
            raise e
