#import standard python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import some basic models, includes linear regression and random forrest
import basic_models
import data_fetch

#function to convert df_target to binary varible
#r_thresh is return threshold, should always be greater than 1
#r_thresh is minimum demanded monthly return to enter position
def convert_binary( x ):
	r_thresh = 1.010

	if x >= r_thresh:
		return 1.0
	else:
		return -1.0

#get all tables, input is target shift and info shift
#target shift denotes what time period to predict return
#info shift is information lag, how long to get new data
info_shift = 7
target_shift = 1*30
df , df_target  =  data_fetch.load_all_tables( fund_id = 1,  info_shift = info_shift , target_shift = target_shift  )
df_target = df_target.applymap( convert_binary  )['target_return']


#data shape
p = len( df.columns )
N = len( df )
print("Number of Features:" , p )
print("Number of Datapoints:" , N )

### split dataset into test and train data
### f_factor is test train split proportions
f_factor = 0.7
df_train , df_test , df_target_train , df_target_test = basic_models.split_data( df , df_target , f_factor )

### scale data, note that fit is done only on train data so no look ahead bias
df_train , df_test  = basic_models.scale_data( df_train , df_test  )

## option to perfrom SVD to reduce dimension, this should increase speed of model fiting
## Avoid if using weekly or monthly data
# N_trunc = 100
# df_train , df_test = basic_models.SVD_truncation( df_train , df_test  , N_trunc )

### as a sanity check, make sure that lenths of data and labels agree
print( "Train Data Loaded Correctly:" , len(df_target_train) == len(df_train) )
print( "Test Data Loaded Correctly:" , len(df_target_test) == len(df_test) )

### fit random forrest classifier on training data
max_depth = 20
clf = basic_models.RFC_fit( df_train , df_target_train , target_shift , max_depth )

### compute predictions and test scores
r_pred_train = clf.predict( df_train )
r_pred_test = clf.predict( df_test )

train_score = clf.score( df_train , df_target_train  )
test_score = clf.score( df_test , df_target_test  )

print("RFC train score:" , train_score )
print("RFC test score:" , test_score)


basic_models.plot_confusion_matrix( df_target_test , r_pred_test , "Test"   )
basic_models.plot_confusion_matrix( df_target_train , r_pred_train , "Train" )

basic_models.plot_ROC( df_target_test , r_pred_test , 'Test'  )
basic_models.plot_ROC( df_target_train , r_pred_train , 'Train'  )


### TO DO:
### 1. Change SQL requests, fit for all funds one at a time
### 2. Get all funds in fun
### 2. Need to add SSE composite returns, compare to backtest performance
### 3. Get return as function of target_shift ie target_shift = 1*30 , 2*30, 3*30
### 4. Put in confidance intervals for correlations, i.e. hypotheses testing


