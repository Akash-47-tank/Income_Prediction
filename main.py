import os
from flask import Flask, request, render_template, Response
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
from wsgiref import simple_server
from ModelTraining import TrainModel
from Training_Validation_Insertation import TrainValidation
from Prediction_From_Model import prediction
import pandas as pd

app = Flask(__name__)
dashboard.bind(app)
app.debug = True
CORS(app)

# Function to run before every request to the Flask application
@app.before_request
def init_measurement():
    pass

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']
            pred_val = TrainValidation(path)  # object initialization
            pred_val.perform_validation()  # calling the prediction_validation function
            pred = prediction(path)  # object initialization
            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % path)
        elif request.form is not None:
            path = request.form['filepath']
            pred_val = prediction(path)  # object initialization
            pred_val.predictionFromModel()  # calling the prediction_validation function
            pred = prediction(path)  # object initialization
            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % path)
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            train_valObj = TrainValidation(path)  # object initialization
            train_valObj.perform_validation()  # calling the training_validation function
            trainModelObj = TrainModel()  # object initialization
            trainModelObj.training_model()  # training the model for the files in the table
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")

port = int(os.getenv("PORT", 5001))
if __name__ == "__main__":
    host = '127.0.0.1'
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
    print("Serving on %s %d" % (host, port))

