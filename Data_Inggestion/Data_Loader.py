import pandas as pd

class DataGetter:
    """
    This class shall be used for obtaining the data from the source for training.
    """
    def __init__(self, data_folder_path, file_object, logger_object):
        self.data_folder_path = data_folder_path
        self.training_file = data_folder_path + "/CensusIncomeData_04072023_110000.csv"
        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self):
        """
        This method reads the data from the source.
        """
        self.logger_object.log(self.file_object, 'Entered the get_data method of the DataGetter class')
        try:
            self.data = pd.read_csv(self.training_file)  # reading the data file
            self.logger_object.log(self.file_object, 'Data Load Successful. Exited the get_data method of the DataGetter class')
            # Returning just a data which we have obtained
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_data method of the DataGetter class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object, 'Data Load Unsuccessful. Exited the get_data method of the DataGetter class')
            raise Exception()
