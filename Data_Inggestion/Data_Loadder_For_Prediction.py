import pandas as pd

class DataGetterPred:
    """
    This class shall be used for obtaining the data from the source for prediction.
    """
    def __init__(self, data_folder_path, file_object, logger_object):
        self.prediction_file = data_folder_path + "/CensusIncomeData_04072023_110000.csv"
        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self):
        """
        This method reads the data from the source.
        """
        self.logger_object.log(self.file_object, 'Entered the get_data method of the DataGetterPred class')
        try:
            self.data = pd.read_csv(self.prediction_file)  # reading the data file
            self.logger_object.log(self.file_object, 'Data Load Successful. Exited the get_data method of the DataGetterPred class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_data method of the DataGetterPred class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object, 'Data Load Unsuccessful. Exited the get_data method of the DataGetterPred class')
            raise Exception()
