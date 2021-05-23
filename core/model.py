#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from maxfw.model import MAXModelWrapper
import pickle  
import pandas as pd
import numpy as np 

import logging
from config import CLASS_MAP

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = {
        'id': 'Churn Prediction',
        'name': 'Prediction Model',
        'description': 'Predict customer will stay or not with the organization',
        'type': 'Sklearn',
        'source': 'Self-managed',
        'license': 'Apache 2.0'
    }

    def __init__(self):
        logger.info('Loading model')

        # Load the model
        with open('files/classifier.pkl', 'rb') as fid:
            self.gnb_loaded = pickle.load(fid)
 
        # Set up instance variables and required inputs for inference

        logger.info('Loaded model')

    def _pre_process(self, inp):
        # define an empty list for adding dummy feature names
        cat_dummies = []
        # open file and read the content in a list
        with open('files/cat_dummies.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]
                # add item to the list
                cat_dummies.append(currentPlace)

        # define an empty list
        processed_columns = []
        # open file and read the content in a list
        with open('files/processed_columns.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]
                # add item to the list
                processed_columns.append(currentPlace)

        return [inp, cat_dummies, processed_columns]

    def _post_process(self, result):
        return [{'prediction': p} for p in [CLASS_MAP[k] for k in result]]

    def _predict(self, x):
        # List of categorical columns
        cat_columns = ['gender', 'SeniorCitizen', 'Partner', 'PhoneService', 
               'MultipleLines', 'InternetService', 'OnlineSecurity', 
               'OnlineBackup', 'DeviceProtection',  'TechSupport',  
               'StreamingTV',  'StreamingMovies', 'Contract',
               'PaperlessBilling', 'PaymentMethod', 'Dependents']
        
        #df_test = pd.read_csv(x[0])
        df_test = x[0].drop(columns=['Unnamed: 0', 'customerID'])
        df_test_processed = pd.get_dummies(df_test, prefix_sep="__", 
                                   columns=cat_columns)

        # Remove additional columns
        for col in df_test_processed.columns:
            if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in x[1]:
                print("Removing additional feature {}".format(col))
                df_test_processed.drop(col, axis=1, inplace=True)

        for col in x[1]:
            if col not in df_test_processed.columns:
                print("Adding missing feature {}".format(col))
                df_test_processed[col] = 0

        df_test_processed = df_test_processed[x[2]]
        return self.gnb_loaded.predict(df_test_processed)      

