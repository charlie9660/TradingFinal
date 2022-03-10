#import standard python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import sql packages
import credentials
import sqlconnector as sql

#import some basic models, includes linear regression and random forrest
import basic_models
import format_data
import data_fetch

from sklearn.preprocessing import StandardScaler

#get all tables
dX_return , dX_risk  , dX_risk2  , dX_subsidiary  , dX_subsidiary2  , dX_subsidiary3 , dX_nv  = data_fetch.load_all_tables( fund_id = 1 , start_date = '2015-01-01' , end_date = '2021-01-01' )

#get all fund ids in table:
fund_ids = dX_return.fund_id.unique()

#target shift values, about 1,2,3 months
target_shifts = [ 1*30 , 2*30 , 3*30  ]

#fit model for all funds:
N_funds = len( fund_ids )
N_target_shifts = len( target_shifts )
fund_train_scores = np.zeros( [ N_funds , N_target_shifts ] )
fund_test_scores = np.zeros( [ N_funds , N_target_shifts ] )


cnt1 = 0
for fund_id in fund_ids:

	cnt2 = 0
	for target_shift in target_shifts:

		info_shift = 7
		df , df_target = format_data.format_table( fund_id , target_shift , info_shift , dX_return , dX_risk  , dX_risk2  , dX_subsidiary  , dX_subsidiary2  , dX_subsidiary3 , dX_nv )

		p = len( df.columns )
		N = len( df )

		print("Fund Id" , fund_id )
		print("Number of Features:" , p )
		print("Number of Datapoints:" , N )
		

		### split test and train data
		f_factor = 0.7
		df_train , df_test , df_target_train , df_target_test = basic_models.split_data( df , df_target , f_factor )

		#scale data: Note that scaling is only fit on training data
		scaler = StandardScaler()
		scaler.fit(df_train)

		#scale train and test data
		df_train = pd.DataFrame( scaler.fit_transform(df_train) , columns = df_train.columns )
		df_test = pd.DataFrame( scaler.fit_transform(df_test) , columns = df_test.columns )	

		## option to perfrom SVD to reduce dimension, this should increase speed of model fiting
		#N_trunc = 200
		#df_train , df_test = basic_models.SVD_truncation( df_train , df_test  , N_trunc )

		#fit random forrest regressor on train data
		max_depth = 10
		regr = basic_models.RF_fit( df_train , df_target_train , fund_id , target_shift , max_depth )

		r_pred_train = regr.predict(df_train)
		r_pred_test = regr.predict( df_test )

		train_score = regr.score( df_train , df_target_train  )
		test_score = regr.score( df_test , df_target_test  )


		fund_train_scores[cnt1 , cnt2] = train_score
		fund_test_scores[cnt1 ,  cnt2 ] = test_score
		cnt2 = cnt2 + 1

	cnt1 = cnt1 + 1


#save results to file
file = './data/Optimize_Time_data.npy'
np.save(file, [ fund_test_scores , fund_train_scores ] , allow_pickle=True)

file = './data/Optimize_Time_metadata.npy'
np.save(file, [ target_shifts , fund_ids ] , allow_pickle=True)

