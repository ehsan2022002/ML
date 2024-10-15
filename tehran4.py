import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from numpy import where 

import seaborn as sns
sns.set_style("whitegrid")

import time
import warnings
warnings.filterwarnings("ignore")

###-----------------------

df1 = pd.read_csv('housePrice.csv')
df2 = pd.read_csv('Tehran_Aria.csv')

house_price_raw =pd.merge(df1, df2, how='inner', on = 'Address')
house_price_raw= house_price_raw[house_price_raw['Loc'] != 0]

house_price_raw.head()
house_price_raw.info() 

dataset = house_price_raw.copy()

dataset['Area'] = dataset['Area'].str.strip()
dataset['Area']= dataset['Area'].astype(str).str.extract('([1-9][0-9]+)', expand=False)

dataset['Room']= dataset['Room'].astype(str).str.extract('([[0-9])', expand=False)

dataset['Address']= dataset['Address'].astype(str).str.extract('([a-zA-Z0-9]+)', expand=False)

dataset['Price']= dataset['Price'].astype(str).str.extract('([1-9][0-9]+)', expand=False)
dataset['Price'] = dataset['Price'].str.strip()

dataset.dropna(inplace = True)
dataset.reset_index(drop = True, inplace = True)

dataset = dataset.drop(columns = ['Price(USD)'])
boolean_features = ['Parking','Warehouse','Elevator']
dataset[boolean_features] = dataset[boolean_features].astype('int64')

dataset.dropna(inplace = True)    
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

dataset['Area']= dataset['Area'] = pd.to_numeric(dataset['Area'] , errors='coerce')
dataset['Price'] = pd.to_numeric(dataset['Price'] ) 
dataset['Room'] = pd.to_numeric(dataset['Room'] ) 

dataset.head()
dataset.skew()

#corolation matrix
dataset.corr()
f = plt.figure(figsize=(19, 15))
plt.matshow(dataset.corr(), fignum=f.number)
plt.xticks(range(dataset.select_dtypes(['number']).shape[1]), dataset.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(dataset.select_dtypes(['number']).shape[1]), dataset.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.show()


#skewness 

dataset.describe()


###------------------------------------


data = dataset.values[:, :-3] 
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from matplotlib import pyplot


# # perform a box-cox transform of the dataset
# pt = PowerTransformer()
# data = pt.fit_transform(data)
# # # convert the array back to a dataframe
# datasetPT = DataFrame(data)

# datasetPT.columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator']
# datasetPT= pd.concat([datasetPT,dataset['Address']], axis = 1)
# datasetPT= pd.concat([datasetPT,dataset['Price']], axis = 1)
# datasetPT= pd.concat([datasetPT,dataset['Loc']], axis = 1)
# dataset = datasetPT

# # histograms of the variables
dataset.hist()
pyplot.show()
dataset.skew()

###--------------------------------------------------
# def lower_upper(x):
#     Q1 = np.percentile(x, 25)
#     Q3 = np.percentile(x, 75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
    
#     return lower, upper



# lower_area, upper_area = lower_upper(dataset['Area'])
# lower_price, upper_price = lower_upper(dataset['Price'])

# area_outliers = np.where(dataset['Area'] > upper_area)
# price_outliers = np.where(dataset['Price'] > upper_price)
# total_outliers = np.union1d(area_outliers, price_outliers)

###-----------------------------------------------
# from sklearn.svm import OneClassSVM
# df = dataset[["Area", "Price"]]
# model = OneClassSVM(kernel = 'rbf').fit(df)
# y_pred = model.predict(df)
# outlier_index = where(y_pred == -1) 
# # filter outlier values
# outlier_values = df.iloc[outlier_index]

# # visualize outputs
# plt.scatter(dataset["Area"], dataset["Price"])
# plt.scatter(outlier_values["Area"], outlier_values["Price"], c = "r")

# total_outliers = np.asarray(outlier_index )
# total_outliers =total_outliers.flatten()
###----------------------------------

from sklearn.neighbors import LocalOutlierFactor
df = dataset[["Area", "Price"]]
model = LocalOutlierFactor(n_neighbors =20, contamination=.03).fit(df)
y_pred = model.fit_predict(df) 
outlier_index = where(y_pred==-1)
outlier_values = df.iloc[outlier_index]

total_outliers = np.asarray(outlier_index )
total_outliers =total_outliers.flatten()

####-------------------------------------------
house_price = dataset.copy()
house_price.drop(total_outliers, inplace = True)
house_price.reset_index(drop = True, inplace = True)


plt.figure(figsize = (16,8))

plt.subplot(2,1,1)
sns.boxplot(x = house_price['Area'])
plt.subplot(2,1,2)
sns.boxplot(x = house_price['Price'])

house_price.skew()
sns.pairplot(house_price, corner = True)

###############
addres_dummy = pd.get_dummies(house_price['Address'])
house_price_final = house_price.merge(addres_dummy, left_index = True, right_index = True)
house_price_final.drop(columns = 'Address', inplace = True)
house_price_final.drop(columns='Loc',inplace=True)
house_price_final.head(3)




X = house_price_final.drop(columns = 'Price')
y = house_price_final['Price']


####X= X[:,1:]  ##avoid dummy variable trap 
X = X.iloc[:,:-1].values


from imblearn.over_sampling import RandomOverSampler 
oversample = RandomOverSampler(sampling_strategy='auto') 
X_over, y_over = oversample.fit_resample(X, y)
X, y = oversample.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from mlxtend.evaluate import bias_variance_decomp

def parameter_finder (model, parameters):
    
    start = time.time()
    
    grid = GridSearchCV(model, 
                        param_grid = parameters, 
                        refit = True, 
                        cv = KFold(shuffle = True, random_state = 1), 
                        n_jobs = -1)
    grid_fit = grid.fit(X_train, y_train)
    y_pred = grid_fit.predict(X_test)
    
    train_score = round(grid_fit.score(X_train, y_train), 4) 
    test_score = round(grid_fit.score(X_test, y_test), 4)
    
    best_estim=grid.best_estimator_
    
    RMSE =  metrics.mean_squared_error(y_test, y_pred, squared=False) #Root Mean Squared Error (RMSE)
    MAE = metrics.mean_absolute_error(y_test, y_pred) #Mean Absolute Error (MAE)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred) #Mean Absolute Percentage Error (MAPE)
    
    model_name = str(model).split('(')[0]
    
    # mse, bias, variance = bias_variance_decomp(
    #     model, 
    #     X_train, y_train.to_numpy(),
    #     X_test, y_test.to_numpy(), 
    #     loss = 'mse' )
    bias=0
    variance=0
    mse=0
    
    end = time.time()
    
    print("The best parameters for {model_name} model is:" ,grid_fit.best_params_)
    print("--" * 10)
    print("The coefficient of determination (R2 score) in the training set is " , train_score,"for" , model_name ," model.")
    print("The coefficient of determination (R2 score) in the testing set is ", test_score ,"for" , model_name , " model")
    print("RMSE is ", RMSE ,"for" , model_name , " model.")
    print("--" * 10)
    print("MSE is ", mse ,"for" , model_name , " model.")
    print("bias is ", bias ,"for" , model_name , " model.")
    print("bias is ", variance ,"for" , model_name , " model.")
    print("--" * 10)
    print("Runtime of the program is:", end - start)
   
        
    return train_score, test_score, RMSE , MAE , MAPE ,mse,bias,variance



lr = LinearRegression(n_jobs = -1)
lr_train_score, lr_test_score, lr_RMSE ,lr_MAE ,lr_MAPE,lr_mse,lr_bias,lr_variance = parameter_finder(lr, {})


dtr = DecisionTreeRegressor(random_state = 1)
param_dtr = {'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3]}
dtr_train_score, dtr_test_score, dtr_RMSE , dtr_MAE ,dtr_MAPE , dtr_mse,dtr_bias,dtr_variance  = parameter_finder(dtr, param_dtr)


rfr = RandomForestRegressor(random_state = 1, n_jobs = -1)
param_rfr = {'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3]}
rfr_train_score, rfr_test_score, rfr_RMSE , rfr_MAE , rfr_MAPE , rfr_mse,rfr_bias,rfr_variance = parameter_finder(rfr, param_rfr)


knr = KNeighborsRegressor(n_jobs = -1)
param_knr = {'n_neighbors': [5, 10, 15, 20],
            'weights': ['uniform', 'distance']}

knr_train_score, knr_test_score, knr_RMSE , knr_MAE , knr_MAPE ,knr_mse,knr_bias,knr_variance = parameter_finder(knr, param_knr)


models_score = pd.DataFrame({'Training score': [lr_train_score, dtr_train_score, rfr_train_score, knr_train_score],
                             'Testing score': [lr_test_score, dtr_test_score, rfr_test_score, knr_test_score],
                             'RMSE': [lr_RMSE,  dtr_RMSE, rfr_RMSE, knr_RMSE],
                             'MAE': [lr_MAE,  dtr_MAE, rfr_MAE, knr_MAE],
                             'MAPE': [lr_MAPE, dtr_MAPE, rfr_MAPE, knr_MAPE],
                             'MSE':[lr_mse,dtr_mse,rfr_mse,knr_mse],
                             'Bias':[lr_bias,dtr_bias,rfr_bias,knr_bias],
                             'Variance':[lr_variance,dtr_variance,rfr_variance,knr_variance]                            
                             },
                             index = ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor', 'KNeighborsRegressor'])


fig, ax = plt.subplots(figsize=(20,10))
sns.set(style='white')
ax.set_title("Camparison", fontsize = 20)
ax = sns.barplot(x = list(models_score.index), y = models_score['RMSE']/1000000000, alpha = 0.7, palette='Greens_r')
ax.set_ylabel("RMSE", fontsize = 20)
sec_ax = ax.twinx()
sec_ax = sns.lineplot(x = list(models_score.index), y = models_score['Training score'], linewidth = 3, color = 'blue')
sec_ax = sns.scatterplot(x = list(models_score.index), y = models_score['Training score'], s = 200)
sec_ax = sns.lineplot(x = list(models_score.index), y = models_score['Testing score'], linewidth = 3, color = 'red')
sec_ax = sns.scatterplot(x = list(models_score.index), y = models_score['Testing score'], s = 200)
sec_ax.set_ylabel("R2 scores", fontsize = 20)
sec_ax.legend(labels = ['Training score', 'Testing score'], fontsize = 20)
sns.despine(offset = 10)
plt.show()

##-------------------------
