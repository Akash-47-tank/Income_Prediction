import pandas as pd
import numpy as np
from File_Operation import file_Methods
from Data_Processing import processing
from Data_Inggestion import Data_Loadder_For_Prediction
from CassandraDB.app_logging.logger import Applogger
from Validation_Of_Training_Row_Data.RowDataValidation import RawDataValidator

class prediction:

    def __init__(self, path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.logger = Applogger()
        self.pred_data_val = RawDataValidator(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile()  # deletes the existing prediction file from the last run!
            self.logger.log(self.file_object, 'Start of Prediction')
            data_getter = Data_Loadder_For_Prediction.DataGetterPred(self.file_object, self.logger)
            data = data_getter.get_data()

            # code change
            # wafer_names = data['Wafer']
            # data = data.drop(labels=['Wafer'], axis=1)

            preprocessor = processing.Preprocessor(self.file_object, self.logger)
            data = preprocessor.remove_columns(data, ['education'])  # remove the column as it doesn't contribute to prediction.
            data = preprocessor.remove_unwanted_spaces(data)  # remove unwanted spaces from the dataframe
            data.replace('?', np.NaN, inplace=True)  # replacing '?' with NaN values for imputation

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if is_null_present:
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation

            # Proceeding with more data pre-processing steps
            scaled_num_df = preprocessor.scale_numerical_columns(data)
            cat_df = preprocessor.encode_categorical_columns(data)
            X = pd.concat([scaled_num_df, cat_df], axis=1)

            file_loader = file_Methods.File_Operation(self.file_object, self.logger)
            kmeans = file_loader.load_model('KMeans')


            clusters = kmeans.predict(X)  # drops the first column for cluster prediction
            X['clusters'] = clusters
            clusters = X['clusters'].unique()
            predictions = []
            for i in clusters:
                cluster_data = X[X['clusters'] == i]
                cluster_data = cluster_data.drop(['clusters'], axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                result = model.predict(cluster_data)
                for res in result:
                    if res == 0:
                        predictions.append('<=50K')
                    else:
                        predictions.append('>50K')

            final = pd.DataFrame(list(zip(predictions)), columns=['Predictions'])
            path = "Prediction_Output_File/Predictions.csv"
            final.to_csv("Prediction_Output_File/Predictions.csv", header=True, mode='a+')  # appends result to prediction file
            self.logger.log(self.file_object, 'End of Prediction')
        except Exception as ex:
            self.logger.log(self.file_object, 'Error occurred while running the prediction!! Error:: %s' % ex)
            raise ex
        return path
