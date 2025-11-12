# For Part 2 - Step 7
# Use in Azure ML deployment - "Select a scoring script for inferencing"

import os
import logging
import json
import numpy
import joblib


def init():
    """
    init function 
    """
    # load model
    global model
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model', 'random_forest_model.pkl')
    model = joblib.load(model_path)

    logging.info("Initialization complete")


def run(raw_data):
    """
    inference run function
    """
    logging.info("Request Received")

    data = json.loads(raw_data)["data"]

    input_features = []
    for item in data:
        single_user_input = [
            float(item['AccountWeeks']),
            float(item['ContractRenewal']),
            float(item['DataPlan']),
            float(item['DataUsage']),
            float(item['CustServCalls']),
            float(item['DayMins']),
            float(item['DayCalls']),
            float(item['MonthlyCharge']),
            float(item['OverageFee']),
            float(item['RoamMins']),
            float(item['ServiceCallsPerWeek'])  # our added input feature
        ]
        input_features.append(single_user_input)
    input_features = numpy.array(input_features)
    result = model.predict(input_features)

    logging.info("Request Processed")

    return {
        "predictedOutcomes": result.tolist(),
        "inputFeatures": data
    }
