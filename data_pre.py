import numpy as np 
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
import seaborn as sns 
import missingno as msno

pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',100)
import pickle
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

def data_preprocess():
	train = pd.read_csv('data/train.csv')
	test = pd.read_csv('data/test.csv')


	# Combining dataset-----------------------------------------------
	train_y = train['SalePrice']
	data = pd.concat((train,test),sort= False).reset_index(drop=True)
	data.drop(['SalePrice','Id'],axis=1,inplace=True)
	data.rename(columns={'1stFlrSF':'FirstFlrSF','2ndFlrSF':'SecondFlrSF','3SsnPorch':'ThreeSsnPorch'}, inplace=True)


	train_y = np.log(train_y + 1) 

	#Missing Data-------------------------------------------------------
	missing_data = pd.DataFrame(data.isnull().sum()).reset_index()
	missing_data.columns = ['ColumnName','MissingCount']

	missing_data['PercentMissing'] = round(missing_data['MissingCount']/data.shape[0],3)*100
	missing_data =missing_data.sort_values(by = 'MissingCount',ascending = False).reset_index(drop = True)

	

	#Drop the PoolQ,MiscFeature and Alley columns 
	data.drop(['PoolQC','MiscFeature','Alley'],axis=1,inplace=True)
	ffill= list(missing_data.ColumnName[18:34])
	data[ffill] = data[ffill].fillna(method = 'ffill')



	
	col_for_zero = ['Fence','FireplaceQu','GarageFinish','GarageQual','GarageCond','GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','MasVnrType']
	data[col_for_zero] = data[col_for_zero].fillna('None')
	data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].dropna().mean())
	data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].dropna().median())
	data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].dropna().median())



	data['YrBltAndRemod']=data['YearBuilt']+data['YearRemodAdd']
	data['TotalSF']=data['TotalBsmtSF'] + data['FirstFlrSF'] + data['SecondFlrSF']
	data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +data['FirstFlrSF'] + data['SecondFlrSF'])
	data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
	data['Total_porch_sf'] = (data['OpenPorchSF'] + data['ThreeSsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF'])
	data['hasfence'] = data['Fence'].apply(lambda x: 0 if x == 0 else 1).astype(str)
	data['hasmasvnr'] = data['MasVnrArea'].apply(lambda x: 0 if x == 0 else 1).astype(str)
	data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0).astype(str)
	data['has2ndfloor'] = data['SecondFlrSF'].apply(lambda x: 1 if x > 0 else 0).astype(str)
	data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0).astype(str)
	data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0).astype(str)
	data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0).astype(str)
	data['MSSubClass'] = data['MSSubClass'].astype(str)
	data['YrSold'] = data['YrSold'].astype(str)
	data['MoSold'] = data['MoSold'].astype(str)



	num_var=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
	cat_var=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object']]
	
	numerical_data = data[num_var]
	categorical_data = data[cat_var]



	#skew X variables
	skew_data = numerical_data.apply(lambda x: x.skew()).sort_values(ascending=False)


	high_skew = skew_data[skew_data > 0.5]
	skew_index = high_skew.index

	for i in skew_index:
	    data[i] = boxcox1p(data[i], boxcox_normmax(data[i] + 1))


	for c_feature in categorical_data.columns:
   	    categorical_data[c_feature] = categorical_data[c_feature].astype('category')
   	    categorical_data = create_dummies(categorical_data , c_feature)
		 
	numerical_data.drop('PoolArea',axis=1,inplace=True)
	numerical_data = numerical_data.apply(outlier_capp)
	numerical_data['PoolArea'] = data.PoolArea




	final_data = pd.concat([categorical_data,numerical_data,train_y],axis=1)
	final_data.columns= [var.strip().replace('.', '_') for var in final_data.columns]
	final_data.columns= [var.strip().replace('&', '_') for var in final_data.columns]
	final_data.columns= [var.strip().replace(' ', '_') for var in final_data.columns]

	overfit = []
	for i in final_data.columns:
	    counts = final_data[i].value_counts()
	    zeros = counts.iloc[0]
	    if zeros / len(final_data) * 100 > 99.94:
	    	overfit.append(i)


	final_data.drop(overfit,axis=1,inplace=True)


	#splitting the data set into two sets
	final_train = final_data.loc[final_data.SalePrice.isnull()==0]
	final_test = final_data.loc[final_data.SalePrice.isnull()==1]
	final_train = final_train.drop('SalePrice',axis=1)
	final_test = final_test.drop('SalePrice',axis=1)


	train_x = final_train

	test_X = final_test
	return train_x,train_y,test_X


def outlier_capp(x):
    x = x.clip(upper = x.quantile(0.90))
    x = x.clip(lower = x.quantile(0.10))
    return x



def create_dummies(df,colname):
    col_dummies = pd.get_dummies(df[colname],prefix =colname)
    col_dummies.drop(col_dummies.columns[0],axis=1,inplace=True)
    df = pd.concat([df,col_dummies],axis=1)
    df.drop(colname,axis=1,inplace=True)
    return df










