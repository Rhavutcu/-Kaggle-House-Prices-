#!/usr/bin/env python3


import numpy as np 
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from data_pre import data_preprocess


train_x,train_y,test_X = data_preprocess()
	
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5,15.6,15.7,15.8,15.9,16]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,0.0009,0.0010,0.0011,0.0012,0.0013,0.0014,0.0015]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,0.0009,0.0010,0.0011,0.0012,0.0013,0.0014,0.0015]
e_l1ratio = [0.05, 0.15,0.2, 0.25,0.3, 0.35,0.4, 0.45,0.5, 0.55,0.6, 0.65,0.7, 0.75,0.8, 0.85, 0.9, 0.95, 0.99, 1]

k_fold =  KFold(n_splits = 10,shuffle = True,random_state = 21)
	

def rmserror(y,y_pred):
	return np.sqrt(mean_squared_error(y,y_pred))

def cv_rmserror(model,X=train_x):
	 rmserror = np.sqrt(-cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv = k_fold))
	 return rmserror

ridge = make_pipeline(RobustScaler(),RidgeCV(alphas = alphas_alt,cv = k_fold))
lasso = make_pipeline(RobustScaler(),LassoCV(max_iter = 1e7,alphas = alphas2,random_state = 42,cv = k_fold))
elasticnet = make_pipeline(RobustScaler(),ElasticNetCV(max_iter = 1e7,alphas = e_alphas,cv = k_fold,l1_ratio =e_l1ratio))
svr = make_pipeline(RobustScaler(),SVR(C= 20,epsilon = 0.008 , gamma = 0.0003))
gbr = GradientBoostingRegressor(n_estimators=3000,learning_rate= 0.05,max_depth=4,max_features='sqrt',min_samples_leaf = 15,min_samples_split = 10,loss = 'huber',random_state =21)
lightgbm = LGBMRegressor(objective = 'regression',num_leaves = 4,learning_rate = 0.01,n_estimators=5000,max_bin = 200,bagging_fraction = 0.75,bagging_freq = 5,bagging_seed=7,feature_fraction=0.2,feature_fraction_seed = 7,verbose=-1)
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,max_depth=3,min_child_weight=0,gamma = 0,subsample=0.7,colsample_bytree=0.7,objective='reg:linear',nthread=-1,scale_pos_weight=1,seed=27,reg_alpha = 0.00006)



def fit_models(model):

	return model.fit(train_x,train_y)
	
def predict(model,data):	

	return model.predict(data)	
	
		    


a = fit_models(ridge) #example ridge regression

print('Predict submission')
submission = pd.read_csv('data/sample_submission.csv')
submission.iloc[:,1] = np.floor(np.expm1(predict(a,test_X)))




print('Root Mean Score Error score on Train data:(example : RidgeCV) ')    
print(rmserror(train_y,predict(a,train_x)))





