# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:46:47 2019

@author: shara
"""

import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
model = pickle.load(open('datasets/bike-sharing-demand/train_days.pkl', 'rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    
    #get data
    data = request.get_json(force=True)
    
    #convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    
    #predictions
    result = model.predict(data_df)
    
    #send data back to browser
    output = {'results': int(result[0])}
    
    #return data
    return jsonify(result=output)

if __name__=='__main__':
    app.run(port=5000, debug=True)
    