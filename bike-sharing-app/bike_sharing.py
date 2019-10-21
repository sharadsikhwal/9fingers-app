# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:12:32 2019

@author: shara
"""

import pandas as pd

df = pd.read_csv('train.csv')
X = df.drop(['count', 'datetime'], axis=1).values
y = df['count'].values

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeRegressor as DTR
regressor = DTR(random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
regressor.score(X_test, y_test)

from sklearn.ensemble import RandomForestRegressor as RFR
regressor_rfr = RFR(n_estimators=25, random_state=2, n_jobs=-1)
regressor_rfr.fit(X_train, y_train)
y_pred_rfr = regressor_rfr.predict(X_test)

from sklearn.metrics import mean_squared_error as MSE
error = MSE(y_test, y_pred) ** (1/2)
error = MSE(y_test, y_pred_rfr) ** (1/2)

'''data = {'season':[1],	
        'holiday':[0],
        'workingday':[0],
        'weather':[3],
        'temp':[8.96],
        'atemp':[12.765],
        'humidity':[40],
        'windspeed':[7.015],
        'casual':[3],
        'registered':[156]}
df_r = pd.DataFrame(data)
data_predict = regressor.predict(df_r)'''

import pickle
pickle.dump(regressor, open('train_days.pkl', 'wb'))


