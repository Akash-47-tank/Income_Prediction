
 # It is based on EDA :

from sklearn.model_selection import train_test_split

from CassandraDB.app_logging import logger
from Data_Inggestion import Data_Loader
from Data_Inggestion.Data_Loadder_For_Prediction import DataGetterPred
from Data_Inggestion.Data_Loader import DataGetter
from Data_Processing import Code_For_CLuster, processing
from Data_Processing import Code_For_CLuster
from Best_Model_Finder import model_tuner
from File_Operation import file_Methods
from CassandraDB.app_logging.logger import Applogger
import numpy as np
import pandas as pd

# Creating the common Logging object
class TrainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("CassandraDB/Training_Logs/ModelTrainingLog.txt", 'a+')

    def training_model(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from source

            data_folder_path = "/Users/aakash/PycharmProjects/income_ineuron/Training_Raw_files_validated"  # Replace with the path to your data folder
            data_getter = DataGetter.get_data(data_folder_path)
            # Here we have got data-Frame
            data = data_getter.get_data()

            #Performing EDA which we have tried on Jupyter notebook

            """doing the data preprocessing"""

            preprocessor = processing.Preprocessor(self.file_object, self.log_writer)
            data = preprocessor.remove_columns(data, ['education'])  # remove the column as it doesn't contribute to prediction.
            data = preprocessor.remove_unwanted_spaces(data)  # remove unwanted spaces from the dataframe

            data.replace('?', np.NaN, inplace=True)  # replacing '?' with NaN values for imputation

            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='salary')
            # encoding the label column - Performing a mapping to it  -  TARGET COLUMN
            Y = Y.map({'<=50K': 0, '>50K': 1})

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if is_null_present:
                X = preprocessor.impute_missing_values(X, cols_with_missing_values)  # missing value imputation

            # Proceeding with more data pre-processing steps
                # With numerical column we are performing scalling
                # With categorical column we are pwrforming encoding
            scaled_num_df = preprocessor.scale_numerical_columns(X)
            cat_df = preprocessor.encode_categorical_columns(X)
            X = pd.concat([scaled_num_df, cat_df], axis=1)

            """Applying the oversampling approach to handle imbalanced dataset"""
            X, Y = preprocessor.handle_imbalanced_dataset(X, Y)

            """ Applying the clustering approach"""

            kmeans = Code_For_CLuster.KMeansClustering(self.file_object, self.log_writer)  # object initialization.
            # using the elbow plot to find the number of optimum clusters
            number_of_clusters = kmeans.elbow_plot(X)

            # Divide the data into those optimum clusters
                # Also here we will pass OG dataset and will provide numbers of cluster also
            X = kmeans.create_clusters(X, number_of_clusters)

            # create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = Y

            # getting the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data = X[X['Cluster'] == i]  # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3,
                                                                    random_state=355)

                model_finder = model_tuner.model_finder(self.file_object, self.log_writer)  # object initialization

                # getting the best model for each of the clusters
                # It has to paramerters : best_model_name and best_model
                best_model_name, best_model = model_finder.get_best_model_auc(x_train, y_train, x_test, y_test)

                # saving the best model to the directory.
                file_op = file_Methods.File_Operation(self.file_object, self.log_writer)
                save_model = file_op.save_model(best_model, best_model_name + str(i))

            # logging the successful Training
            self.log_writer.log(self.file_object, '-----Successful End of Training-----')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, '-----Unsuccessful End of Training----- ')
            self.file_object.close()
            raise Exception
