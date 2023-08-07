import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Initialize the XGBClassifier with the parameters
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bynode=1, colsample_bytree=1, gamma=0,
                    learning_rate=0.1, max_delta_step=0, max_depth=9,
                    min_child_weight=1, missing=np.nan, n_estimators=130, n_jobs=1,
                    nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    subsample=1, verbosity=1)

# For GridSearchCV - parameter tuning
gnb = GaussianNB(priors=None, var_smoothing=0.05)
prm_grid = {"var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]}


class model_finder:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.logger_object.info("model_finder object created")
        self.xgb = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1)
        self.logger_object.info("model_finder object created")



    def get_best_params_for_logistic_regression(self, train_x, train_y):
        """
        This function finds the model with the best accuracy and AUC score for Logistic Regression.

        Parameters:
        train_x (array-like): The feature matrix of the training data.
        train_y (array-like): The target vector of the training data.

        Returns:
        best_params (dict): Dictionary containing the best hyperparameters for the model.
        """
        try:
            self.logger_object.info("Starting the model selection process for Logistic Regression.")

            # Define the hyperparameter grid to search over
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }

            # Create a logistic regression model
            model = LogisticRegression()

            # Perform grid search using cross-validation to find the best hyperparameters
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                       cv=5, n_jobs=-1, scoring='roc_auc')
            grid_search.fit(train_x, train_y)

            # Get the best hyperparameters and corresponding scores
            best_params = grid_search.best_params_
            best_accuracy = grid_search.best_score_
            best_auc_score = roc_auc_score(train_y, grid_search.predict_proba(train_x)[:, 1])

            self.logger_object.info("Model selection completed for Logistic Regression.")
            self.logger_object.info(f"Best parameters: {best_params}")
            self.logger_object.info(f"Best accuracy: {best_accuracy:.3f}")
            self.logger_object.info(f"Best AUC score: {best_auc_score:.3f}")

            return best_params

        except Exception as e:
            self.logger_object.exception(f"Exception occurred: {str(e)}")
            raise

    def  get_best_params_for_XGBoost(self, train_x, train_y):
        """
        Method Name: get_best_params_for_XGBoost
        Description: Get the parameters for XGBoost Algorithm which give the best accuracy.
                     Use Hyperparameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        """
        try:
            self.logger_object.info("Entered the get_best_params_for_XGBoost method of the Model_Finder class")

            # Define the hyperparameter grid to search over
            param_grid_xgboost = {
                "n_estimators": [100, 130],
                "criterion": ['gini', 'entropy'],
                "max_depth": range(8, 10, 1)
            }

            # Create an XGBoost model
            model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1)

            # Perform grid search using cross-validation to find the best hyperparameters
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid_xgboost,
                                       verbose=3, cv=5, scoring='roc_auc')
            grid_search.fit(train_x, train_y)

            # Get the best hyperparameters and corresponding scores
            best_params = grid_search.best_params_
            best_accuracy = grid_search.best_score_
            best_auc_score = roc_auc_score(train_y, grid_search.predict_proba(train_x)[:, 1])

            # Create a new model with the best parameters
            best_model = xgb.XGBClassifier(
                n_estimators=best_params['n_estimators'],
                criterion=best_params['criterion'],
                max_depth=best_params['max_depth'],
                objective='binary:logistic',
                n_jobs=-1
            )

            # Train the new model with the best parameters
            best_model.fit(train_x, train_y)

            self.logger_object.info("XGBoost best params: " + str(best_params))
            self.logger_object.info("Best accuracy: %.3f", best_accuracy)
            self.logger_object.info("Best AUC score: %.3f", best_auc_score)
            self.logger_object.info("Exited the get_best_params_for_XGBoost method of the Model_Finder class")

            return best_model

        except Exception as e:
            self.logger_object.exception(
                "Exception occurred in get_best_params_for_XGBoost method of the Model_Finder class. Exception message: %s",
                str(e))
            self.logger_object.error(
                "XGBoost Parameter tuning failed. Exited the get_best_params_for_XGBoost method of the Model_Finder class")
            raise Exception("XGBoost Parameter tuning failed.")

    def get_best_params_for_Naive_Bayes(self, train_x, train_y):
        """
        Method Name: get_best_params_for_Naive_Bayes
        Description: Get the parameters for Gaussian Naive Bayes Algorithm which give the best accuracy.
                     Use Hyperparameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception
        """
        try:
            self.logger_object.info("Entered the get_best_params_for_Naive_Bayes method of the Model_Finder class")

            # Define the hyperparameter grid to search over
            param_grid_gnb = {
                'var_smoothing': np.logspace(-9, 0, 10)
            }

            # Create a Gaussian Naive Bayes model
            model = GaussianNB()

            # Perform grid search using cross-validation to find the best hyperparameters
            grid_search = GridSearchCV(estimator=gnb, param_grid=prm_grid,
                                       verbose=3, cv=5, scoring='accuracy')
            grid_search.fit(train_x, train_y)

            # Get the best hyperparameters and corresponding accuracy score
            best_params = grid_search.best_params_
            best_accuracy = grid_search.best_score_

            # Create a new model with the best parameters
            best_model = GaussianNB(var_smoothing=best_params['var_smoothing'])

            # Train the new model with the best parameters
            best_model.fit(train_x, train_y)

            self.logger_object.info("Gaussian Naive Bayes best params: " + str(best_params))
            self.logger_object.info("Best accuracy: %.3f", best_accuracy)
            self.logger_object.info("Exited the get_best_params_for_Naive_Bayes method of the Model_Finder class")

            return best_model

        except Exception as e:
            self.logger_object.exception(
                "Exception occurred in get_best_params_for_Naive_Bayes method of the Model_Finder class. Exception message: %s",
                str(e))
            self.logger_object.error(
                "Gaussian Naive Bayes Parameter tuning failed. Exited the get_best_params_for_Naive_Bayes method of the Model_Finder class")
            raise Exception("Gaussian Naive Bayes Parameter tuning failed.")

    def get_best_model_auc(self, train_x, train_y, test_x, test_y):
        """
        Find out the Model which has the best AUC score among XGBoost, Logistic Regression, and Naive Bayes.

        Parameters:
        train_x (array-like): The feature matrix of the training data.
        train_y (array-like): The target vector of the training data.
        test_x (array-like): The feature matrix of the testing data.
        test_y (array-like): The target vector of the testing data.

        Returns:
        best_model_name (str): The name of the best model (XGBoost, Logistic Regression, or Naive Bayes).
        best_model (object): The best model object with the best parameters.
        """
        try:
            self.logger_object.info("Entered the get_best_model_auc method of the Model_Finder class")

            # Get the best model for XGBoost
            xgboost_model = self.get_best_params_for_XGBoost(train_x, train_y)
            xgboost_predictions = xgboost_model.predict(test_x)

            if len(test_y.unique()) == 1:
                xgboost_auc_score = accuracy_score(test_y, xgboost_predictions)
                self.logger_object.info(f'Accuracy for XGBoost: {xgboost_auc_score:.3f}')
            else:
                xgboost_auc_score = roc_auc_score(test_y, xgboost_predictions)
                self.logger_object.info(f'AUC for XGBoost: {xgboost_auc_score:.3f}')

            # Get the best model for Logistic Regression
            logistic_regression_model = self.get_best_params_for_logistic_regression(train_x, train_y)
            logistic_regression_predictions = logistic_regression_model.predict(test_x)

            if len(test_y.unique()) == 1:
                logistic_regression_auc_score = accuracy_score(test_y, logistic_regression_predictions)
                self.logger_object.info(f'Accuracy for Logistic Regression: {logistic_regression_auc_score:.3f}')
            else:
                logistic_regression_auc_score = roc_auc_score(test_y, logistic_regression_predictions)
                self.logger_object.info(f'AUC for Logistic Regression: {logistic_regression_auc_score:.3f}')

            # Get the best model for Naive Bayes
            naive_bayes_model = self.get_best_params_for_Naive_Bayes(train_x, train_y)
            naive_bayes_predictions = naive_bayes_model.predict(test_x)

            if len(test_y.unique()) == 1:
                naive_bayes_auc_score = accuracy_score(test_y, naive_bayes_predictions)
                self.logger_object.info(f'Accuracy for Naive Bayes: {naive_bayes_auc_score:.3f}')
            else:
                naive_bayes_auc_score = roc_auc_score(test_y, naive_bayes_predictions)
                self.logger_object.info(f'AUC for Naive Bayes: {naive_bayes_auc_score:.3f}')

            # Compare the AUC scores of the models
            models = {
                'XGBoost': xgboost_auc_score,
                'Logistic Regression': logistic_regression_auc_score,
                'Naive Bayes': naive_bayes_auc_score
            }
            best_model_name = max(models, key=models.get)
            best_model = xgboost_model if best_model_name == 'XGBoost' else (
                logistic_regression_model if best_model_name == 'Logistic Regression' else naive_bayes_model
            )

            self.logger_object.info(f'Best Model: {best_model_name} with AUC score: {models[best_model_name]:.3f}')
            self.logger_object.info("Exited the get_best_model_auc method of the Model_Finder class")

            return best_model_name, best_model

        except Exception as e:
            self.logger_object.exception(f"Exception occurred: {str(e)}")
            raise