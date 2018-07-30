#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The reason behind variables selection is based on 
1.Stability 
2. Ability to pay 
3. Willing to pay

Flow :
Part 1 :application_train /application_test 
Part 2 :bureau_balance
PArt 3 :bureau
Part 4 :Previous application
Part 5 :Credit_card_balance
Part 6 :POS_CASH_balance 
Part 7 :Installments_payments

General ideas : 
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Certain features are created by ratio 
# All tables are joined with the application (main table) using the SK_ID_CURR key (except bureau_balance).
"""


import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score, train_test_split
pd.options.display.float_format = "{:.4f}".format


## Load data  
app_train=pd.read_csv('/Users/mirabooboo/Desktop/kaggle_project2_home_credit/application_train.csv')
app_test = pd.read_csv('/Users/mirabooboo/Desktop/kaggle_project2_home_credit/application_test.csv')
bureau = pd.read_csv('/Users/mirabooboo/Desktop/kaggle_project2_home_credit/bureau.csv')
bureau_bal = pd.read_csv('/Users/mirabooboo/Desktop/kaggle_project2_home_credit/bureau_balance.csv')
POS_CASH_bal = pd.read_csv('/Users/mirabooboo/Desktop/kaggle_project2_home_credit/POS_CASH_balance.csv')
previous_app = pd.read_csv('/Users/mirabooboo/Desktop/kaggle_project2_home_credit/previous_application.csv')
instal_pay = pd.read_csv('/Users/mirabooboo/Desktop/kaggle_project2_home_credit/installments_payments.csv')
credit_card_bal = pd.read_csv('/Users/mirabooboo/Desktop/kaggle_project2_home_credit/credit_card_balance.csv')
print('Size of application_train data', app_train.shape)
print('Size of application_test data', app_test.shape)
print('Size of bureau data', bureau.shape)
print('Size of bureau_balance data', bureau_bal.shape)
print('Size of POS_CASH_balance data', POS_CASH_bal.shape)
print('Size of previous_application data', previous_app.shape)
print('Size of installments_payments data', instal_pay .shape)
print('Size of credit_card_balance data', credit_card_bal.shape)

"""
Part 1 : Application dataset


"""

### check datatype
numerical = [i for i in app_train.columns if app_train.dtypes[i] != 'object']
categorical = [i for i in app_train.columns if app_train.dtypes[i] == 'object']
print("numerical: {}, categorical: {}" .format (len(numerical),len(categorical)))

### Check target variable -highly imbalance
total = app_train['TARGET'].value_counts()
percent = (app_train['TARGET'].value_counts()/len(app_train)*100)
target_var  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(target_var)

### combine training and test dataset
app_test['TARGET'] = 2
frames =[app_train, app_test]
app_data = pd.concat(frames)
app_data.shape


### check missing value 
total = app_data.isnull().sum().sort_values(ascending = False)
percent = (app_data.isnull().sum()/app_data.isnull().count()*100).sort_values(ascending = False)
missing_app_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_app_data.head(30)


def data_preprocessing(dataframe):
    df = dataframe
    ## drop  irreverent columns
    cols = (['COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE',
            'FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE',
            'LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','ENTRANCES_AVG',
            'NONLIVINGAREA_MODE','COMMONAREA_MEDI','ELEVATORS_MEDI',
            'ENTRANCES_MEDI','FLOORSMAX_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI',
            'LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI',
            'NONLIVINGAREA_AVG','COMMONAREA_AVG','LIVINGAPARTMENTS_AVG', 'WALLSMATERIAL_MODE',
            'NONLIVINGAPARTMENTS_AVG','FONDKAPREMONT_MODE','EMERGENCYSTATE_MODE',
            'BASEMENTAREA_AVG','BASEMENTAREA_MEDI', 'BASEMENTAREA_MODE',
             'ELEVATORS_AVG'])
    for i in cols:
        df.drop(i, axis=1, inplace=True)
    
    # Optional: Remove 4 applications with unknown GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    ##Binary encode
    
    #Recode Contract Types
    df.NAME_CONTRACT_TYPE.replace('Cash loans', 0, inplace=True)
    df.NAME_CONTRACT_TYPE.replace('Revolving loans', 1, inplace=True)

    #Recode Gender
    df.CODE_GENDER.replace('M', 0, inplace=True)
    df.CODE_GENDER.replace('F', 1, inplace=True)
    #Recode Own Car
    df.FLAG_OWN_CAR.replace('Y', 1, inplace=True)
    df.FLAG_OWN_CAR.replace('N', 0, inplace=True)
    #Record Own Realty
    df.FLAG_OWN_REALTY.replace('Y', 1, inplace=True)
    df.FLAG_OWN_REALTY.replace('N', 0, inplace=True)
    #Recode Contract Types
    df.NAME_CONTRACT_TYPE.replace('Cash loans', 0, inplace=True)
    df.NAME_CONTRACT_TYPE.replace('Revolving loans', 1, inplace=True)
        
# #     # Create an anomalous flag column
#   df['DAYS_EMPLOYED_Anomaly'] = df[df["DAYS_EMPLOYED"] == 365243]
    
    # covert days to year
    df['age'] = df['DAYS_BIRTH'] / -365
    df['Years_employed'] = df['DAYS_EMPLOYED'] / -365
    df['yr_ID_PUBLISH']=df['DAYS_ID_PUBLISH']/-365
    df['yr_PHONE_CHANGE']=df['DAYS_LAST_PHONE_CHANGE']/-365
    
    # Replace the anomalous values with nan
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    
    df['DAYS_LAST_PHONE_CHANGE'].replace(0,np.nan,inplace = True)
    
    df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)
    return  df

df=data_preprocessing(app_data)

df.head()


### Feature engineering
df['AMT_INCOME_MONTHLY']=df['AMT_INCOME_TOTAL']/12
df['CREDIT_INCOME_Ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_INCOME_Ratio'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['PAYMENT_Ratio'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
df['# of contact cnt'] = df['FLAG_MOBIL'] + df['FLAG_EMP_PHONE'] + df['FLAG_WORK_PHONE'] + df['FLAG_CONT_MOBILE'] + df['FLAG_PHONE'] + df['FLAG_EMAIL']
df['AMT_INCOME_TOTAL'] = np.log1p(df['AMT_INCOME_TOTAL'])
df['AMT_CREDIT'] = np.log1p(df['AMT_CREDIT'])
df['AMT_ANNUITY'] = np.log1p(df['AMT_ANNUITY'])
df['AMT_GOODS_PRICE'] = np.log1p(df['AMT_GOODS_PRICE'])

### drop columns for feature engineering
df = df.drop(['DAYS_EMPLOYED','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL',
                                  'DAYS_BIRTH','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE'], axis=1)
    
    
df1 = pd.get_dummies(df)
print(df1.shape) 

##################################################################
### Baseline model : Light XGB
df_train = df1[df1['TARGET']!=2]
df_test = df1[df1['TARGET']==2]
df_test = df_test.drop(['TARGET'], axis=1)
y = df_train['TARGET']
X = df_train.drop(['TARGET','SK_ID_CURR'], axis=1)
print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)
print('X: ', X.shape)
print('y: ', y.shape)
   

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
lgb_train = lgb.Dataset(data=X_train, label=y_train)
lgb_eval = lgb.Dataset(data=X_valid, label=y_valid)


params = {'task': 'train', 
          'boosting_type': 'gbdt', 
          'objective': 'binary', 
          'metric': 'auc', 
          'learning_rate': 0.01, 
          'num_leaves': 48, 
          'num_iteration': 5000, 
          'verbose': 0 ,
          'colsample_bytree':.8, 
          'subsample':.9, 
          'max_depth':7, 
          'reg_alpha':.1, 
          'reg_lambda':.1, 
          'min_split_gain':.01, 
          'min_child_weight':1}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
lgb.plot_importance(model, figsize=(10, 14), color='skyblue', max_num_features=50);

######################################################################

"""
Part 2 : Bureau balance data    
-Monthly data 
-Status of Credit Bureau loan during the month (active, closed, DPD0-30,Ö [C means closed, X means status unknown, 0 means no DPD, 
1 means maximal did during month between 1-30, 2 means DPD 31-60,Ö 5 means DPD 120+ or sold or written off ] )
"""

# map the status, DPD>1 consider as 1 else 0
def partition(x):
    if x =='C':
        return '0'
    elif x=='0':
        return '0'
    elif x=='X' :
        return '0'
    return '1'

bureau_bal['STATUS1'] = bureau_bal['STATUS'].map(partition)
bureau_bal_agg1= bureau_bal.groupby(['SK_ID_BUREAU']).size().reset_index(name='tot_count')
bureau_bal_agg2= bureau_bal[bureau_bal['STATUS1'] == '1'].groupby(['SK_ID_BUREAU']).size().reset_index(name='late_count')
bureau_bal_agg3 = bureau_bal_agg1.merge(bureau_bal_agg2, on=['SK_ID_BUREAU'], how='left')
bureau_bal_agg3['late_ratio']=bureau_bal_agg3['late_count']/bureau_bal_agg3['tot_count']
bureau_bal_agg3['late_ratio'] = bureau_bal_agg3['late_ratio'].fillna(0)

# merge with Bureau data 
bureau_bal_final = bureau_bal_agg3[['SK_ID_BUREAU','late_ratio']]
bureau= bureau.merge(bureau_bal_final, on = 'SK_ID_BUREAU', how = 'left')

######################################################################

"""
Part 3 : Bureau data    

"""

# total count of loans in bureau
bureau_agg = pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].unique()})
bureau_agg['loans_count'] = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].transform('count')

# consider active status feature 
active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
active1= active.groupby(['SK_ID_CURR']).size().reset_index(name='Active_app_count')
active2= active.groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index(name='Active_app_credit_amt') 


# Group by the client id, calculate aggregation statistics
bureau_agg1 = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

# List of column names
columns = ['SK_ID_CURR']

# Iterate through the variables names
for var in bureau_agg1.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in bureau_agg1.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('bureau_%s_%s' % (var, stat))
            
            
# Assign the list of columns names as the dataframe column names
bureau_agg1.columns = columns
bureau_agg1.head()            
 
           
# merge back to bureau dataset 
bureau_agg1 = bureau_agg1.merge(bureau_agg, on=['SK_ID_CURR'], how='left')
bureau_agg1 = bureau_agg1.merge(active1, on=['SK_ID_CURR'], how='left')
bureau_agg1 = bureau_agg1.merge(active2, on=['SK_ID_CURR'], how='left')

            
# merge back to main dataset & create agg_level feature 
df1= df1.merge(bureau_agg1, on = 'SK_ID_CURR', how = 'left')
df1['AMT_CREDIT_debit_ratio']=df1['bureau_AMT_CREDIT_SUM_DEBT_sum']/df1['bureau_AMT_CREDIT_SUM_sum'] 
df1['active_ratio']=df1['Active_app_count']/df1['loans_count']


"""
Part 4 : Previous application  

"""

### replace weired number 
previous_app['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
previous_app['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
previous_app['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
previous_app['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
previous_app['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

# Group by the client id, calculate aggregation statistics
previous_app_agg = previous_app.drop(columns = ['SK_ID_PREV']).groupby('SK_ID_CURR', as_index = False).agg(['var', 'mean', 'max', 'min', 'sum']).reset_index()

# List of column names
columns = ['SK_ID_CURR']

# Iterate through the variables names
for var in previous_app_agg.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in previous_app_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('previous_%s_%s' % (var, stat))

previous_app_agg.columns = columns

# merge back to main dataset & create agg_level feature 
df1= df1.merge(previous_app_agg, on = 'SK_ID_CURR', how = 'left')
df1['previous_APP_CREDIT_ratio'] = df1['previous_AMT_APPLICATION_sum'] / df1['previous_AMT_CREDIT_sum']



"""
Part 5 : credit card balance dataset (monthly) 

"""
# Group by the client id, calculate aggregation statistics
credit_card_agg = credit_card_bal.drop(columns = ['SK_ID_PREV']).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max', 'min', 'sum']).reset_index()

# List of column names
columns = ['SK_ID_CURR']

# Iterate through the variables names
for var in credit_card_agg.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in credit_card_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('creditcrd_%s_%s' % (var, stat))
            
credit_card_agg.columns = columns         
   
# merge back to main dataset 
df1= df1.merge(credit_card_agg, on = 'SK_ID_CURR', how = 'left')

"""
Part 6: credit card balance dataset (monthly) 

"""
# Group by the client id, calculate aggregation statistics
POS_CASH_agg = POS_CASH_bal.drop(columns = ['SK_ID_PREV']).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max','sum']).reset_index()


# List of column names
columns = ['SK_ID_CURR']

# Iterate through the variables names
for var in POS_CASH_agg.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in POS_CASH_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('POS_%s_%s' % (var, stat))
            
POS_CASH_agg.columns = columns

# merge back to main dataset 
df1= df1.merge(POS_CASH_agg, on = 'SK_ID_CURR', how = 'left')            


"""
Part 7: instalallement 

"""
# Group by the client id, calculate aggregation statistics
instal_agg = instal_pay.drop(columns = ['SK_ID_PREV']).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max','sum']).reset_index()


# List of column names
columns = ['SK_ID_CURR']

# Iterate through the variables names
for var in instal_agg.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in instal_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('instal_%s_%s' % (var, stat))

instal_agg.columns = columns

# merge back to main dataset 
df1= df1.merge(instal_agg, on = 'SK_ID_CURR', how = 'left')
df1['inatal_PAYMENT_Ratio']= df1['instal_AMT_PAYMENT_sum']/df1['instal_AMT_INSTALMENT_sum']

print(df1.shape)


### model 2 
df2=df1.copy()
df_train2 = df2[df2['TARGET']!=2]
df_test2 = df2[df2['TARGET']==2]
df_test2 = df_test2.drop(['TARGET'], axis=1)
y2 = df_train2['TARGET']
X2 = df_train2.drop(['TARGET','SK_ID_CURR'], axis=1)
print('Training Features shape: ', df_train2.shape)
print('Testing Features shape: ', df_test2.shape)
print('X2: ', X2.shape)
print('y2: ', y2.shape)

X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X2, y2, test_size=0.20, random_state=42)
lgb_train2 = lgb.Dataset(data=X_train2, label=y_train2)
lgb_eval2 = lgb.Dataset(data=X_valid2, label=y_valid2)

params = {'task': 'train', 
          'boosting_type': 'gbdt', 
          'objective': 'binary', 
          'metric': 'auc', 
          'learning_rate': 0.01, 
          'num_leaves': 48, 
          'num_iteration': 5000, 
          'verbose': 0 ,
          'colsample_bytree':.8, 
          'subsample':.9, 
          'max_depth':7, 
          'reg_alpha':.1, 
          'reg_lambda':.1, 
          'min_split_gain':.01, 
          'min_child_weight':1}
model = lgb.train(params, lgb_train2, valid_sets=lgb_eval2, early_stopping_rounds=150, verbose_eval=200)

lgb.plot_importance(model, figsize=(10, 14), color='skyblue', max_num_features=80);

##################################################################

### model 3-drop top 30 missing variables  ####
df3=df2.copy()
df3 = df3.drop([
'previous_DAYS_FIRST_DRAWING_var' ,             
'previous_RATE_INTEREST_PRIMARY_var' ,          
'previous_RATE_INTEREST_PRIVILEGED_var',        
'previous_RATE_INTEREST_PRIVILEGED_max',        
'previous_RATE_INTEREST_PRIVILEGED_min' ,       
'previous_RATE_INTEREST_PRIMARY_mean' ,        
'previous_RATE_INTEREST_PRIMARY_max' ,          
'previous_RATE_INTEREST_PRIMARY_min' ,          
'previous_RATE_INTEREST_PRIVILEGED_mean',       
'previous_DAYS_FIRST_DRAWING_min',             
'previous_DAYS_FIRST_DRAWING_max',             
'previous_DAYS_FIRST_DRAWING_mean',             
'creditcrd_AMT_PAYMENT_CURRENT_min',           
'creditcrd_AMT_PAYMENT_CURRENT_max',            
'creditcrd_AMT_PAYMENT_CURRENT_mean',           
'creditcrd_AMT_DRAWINGS_POS_CURRENT_min',       
'creditcrd_CNT_DRAWINGS_ATM_CURRENT_min',      
'creditcrd_CNT_DRAWINGS_POS_CURRENT_min',       
'creditcrd_CNT_DRAWINGS_POS_CURRENT_max',       
'creditcrd_AMT_DRAWINGS_ATM_CURRENT_min',       
'creditcrd_CNT_DRAWINGS_OTHER_CURRENT_min',     
'creditcrd_CNT_DRAWINGS_OTHER_CURRENT_max',     
'creditcrd_CNT_DRAWINGS_OTHER_CURRENT_mean',    
'creditcrd_CNT_DRAWINGS_ATM_CURRENT_max',       
'creditcrd_AMT_DRAWINGS_ATM_CURRENT_mean',      
'creditcrd_CNT_DRAWINGS_ATM_CURRENT_mean',      
'creditcrd_AMT_DRAWINGS_OTHER_CURRENT_mean',  
'creditcrd_AMT_DRAWINGS_OTHER_CURRENT_max',     
'creditcrd_AMT_DRAWINGS_OTHER_CURRENT_min',     
'creditcrd_AMT_DRAWINGS_POS_CURRENT_mean'],axis=1)
    
    
df_train3 = df3[df3['TARGET']!=2]
df_test3 = df3[df3['TARGET']==2]
df_test3 = df_test3.drop(['TARGET'], axis=1)
y3 = df_train3['TARGET']
X3 = df_train3.drop(['TARGET','SK_ID_CURR'], axis=1)
print('Training Features shape: ', df_train3.shape)
print('Testing Features shape: ', df_test3.shape)
print('X3: ', X3.shape)
print('y3: ', y3.shape)

X_train3, X_valid3, y_train3, y_valid3 = train_test_split(X3, y3, test_size=0.20, random_state=42)
lgb_train3 = lgb.Dataset(data=X_train3, label=y_train3)
lgb_eval3 = lgb.Dataset(data=X_valid3, label=y_valid3)

params = {'task': 'train', 
          'boosting_type': 'gbdt', 
          'objective': 'binary', 
          'metric': 'auc', 
          'learning_rate': 0.01, 
          'num_leaves': 48, 
          'num_iteration': 5000, 
          'verbose': 0 ,
          'colsample_bytree':.8, 
          'subsample':.9, 
          'max_depth':7, 
          'reg_alpha':.1, 
          'reg_lambda':.1, 
          'min_split_gain':.01, 
          'min_child_weight':1}
model = lgb.train(params, lgb_train3, valid_sets=lgb_eval3, early_stopping_rounds=150, verbose_eval=200)

lgb.plot_importance(model, figsize=(10, 14), color='skyblue', max_num_features=80);    


##################################################################

### Model 4-try upsampling ####


# perform upsampling to address sample imbalance
from sklearn.utils import resample
# Separate majority and minority classes
train_positive= df3[df3['TARGET']==0]
train_negative = df3[df3['TARGET']==1]

# Upsample minority class
train_negative_upsample = resample(train_negative, 
                                 replace=True,     # sample with replacement
                                 n_samples=74475,    # Upsample 3X (24825*3) to match majority class
                                 random_state=42) # reproducible results
 
# Combine majority class with upsampled minority class
train_upsample = pd.concat([train_positive, train_negative_upsample])
 
# Display new class counts
print(train_upsample['TARGET'].value_counts()/len(train_upsample))

df_test4 = df3[df3['TARGET']==2]
df_test4 = df_test4.drop(['TARGET'], axis=1)
y4 = train_upsample['TARGET']
X4 = train_upsample.drop(['TARGET','SK_ID_CURR'], axis=1)
print('X4: ', X4.shape)
print('y4: ', y4.shape)

X_train4, X_valid4, y_train4, y_valid4 = train_test_split(X4, y4, test_size=0.20, random_state=42)
lgb_train4 = lgb.Dataset(data=X_train4, label=y_train4)
lgb_eval4 = lgb.Dataset(data=X_valid4, label=y_valid4)

params = {'task': 'train', 
          'boosting_type': 'gbdt', 
          'objective': 'binary', 
          'metric': 'auc', 
          'learning_rate': 0.01, 
          'num_leaves': 48, 
          'num_iteration': 5000, 
          'verbose': 0 ,
          'colsample_bytree':.8, 
          'subsample':.9, 
          'max_depth':7, 
          'reg_alpha':.1, 
          'reg_lambda':.1, 
          'min_split_gain':.01, 
          'min_child_weight':1}
model = lgb.train(params, lgb_train4, valid_sets=lgb_eval4, early_stopping_rounds=150, verbose_eval=200)

lgb.plot_importance(model, figsize=(10, 14), color='skyblue', max_num_features=50);

