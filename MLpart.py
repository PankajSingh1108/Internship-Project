import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression


df = pd.read_csv('flight_data_clean.csv')
df = df.iloc[: , 1:]
df.head()

df.rename(columns = {'Go Air':'GoAir'}, inplace = True)

df.columns

cols = ['Dept_city', 'Dept_date', 'arrival_city', 'stops','Dep_hour', 'Dep_min',
       'Arrival_hour', 'Arrival_min', 'Duration_hour', 'Duration_min',
       'AirAsia', 'GoAir', 'IndiGo', 'Spicejet', 'Vistara', 'E', 'PE']
label = ['optimal_hours', 'Price']

# 'B' and 'Air India' are 0 in OHE

X = df[cols].copy()
y = df[label].copy()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)
print("X_train shape is:",X_train.shape)
print("X_test shape is:",X_test.shape)
print("y_train shape is:",y_train.shape)
print("y_test shape is:",y_test.shape)
#####################################################

from sklearn.ensemble import RandomForestRegressor
RandomForestRegressorModel = RandomForestRegressor(max_depth=10,random_state=0,n_jobs=-1)#n_estimators=10000,max_depth=10, n_jobs=-1,
RandomForestRegressorModel.fit(X_train, y_train)

y_pred = RandomForestRegressorModel.predict(X_test)

print('Random Forest Regressor Train Score is : ' , RandomForestRegressorModel.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , RandomForestRegressorModel.score(X_test, y_test))
print('Random Forest Regressor No. of features are : ' , RandomForestRegressorModel.n_features_)
print('----------------------------------------------------')

print('Predicted Value for Random Forest Regressor is : ' , y_pred[:49])
print("score of model is :",RandomForestRegressorModel.score(X_test,y_test))

################################################### XGB regressor
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

xgb = XGBRegressor(learning_rate = 0.09,n_jobs = -1)
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)

print('xgbregressor Train Score is : ' , xgb.score(X_train, y_train))
print('xgbregressor Test Score is : ' , xgb.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(xgb, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
print('R2 score:', r2_score(y_test, y_pred_xgb))

#################################### XGB with Multioutput
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
estimator = XGBRegressor(learning_rate = 0.2, max_depth = 9)

# Define the model
xgbregressor = MultiOutputRegressor(estimator = estimator, n_jobs = -1).fit(X_train, y_train)
y_pred_xgb = xgbregressor.predict(X_test)

print('xgbregressor Train Score is : ' , xgbregressor.score(X_train, y_train))
print('xgbregressor Test Score is : ' , xgbregressor.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(xgbregressor, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
print('R2 score:', r2_score(y_test, y_pred_xgb))


#################################### Run and saves Xgboost regressor with 1 output
import pickle
# open a file, where you ant to store the data
file = open('xgboost.pkl', 'wb')

# dump information to that file
pickle.dump(xgb, file)
model = open('xgboost.pkl','rb')
xgboostreg = pickle.load(model)
y_prediction = xgboostreg.predict(X_test)
print(r2_score(y_test, y_prediction))

######################################## Saves Multi output model

import pickle
# open a file, where you ant to store the data
file = open('xgboostregressor.pkl', 'wb')

# dump information to that file
pickle.dump(xgbregressor, file)

#####predictions
model = open('xgboostregressor.pkl','rb')
xgboostregmo = pickle.load(model)
y_prediction = xgboostregmo.predict(X_test)
print(r2_score(y_test, y_prediction))






