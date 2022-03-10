#import basic python packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import basic models
from sklearn.linear_model import Ridge, Lasso ,  LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import joblib


#split data into train and test sets
def split_data(df , df_target , f_factor ):
	
	N = len( df )

	#using a 70-30 train-test split
	index = int( f_factor * N  )

	#train data
	df_train = df[0:index]
	df_target_train = df_target[0:index]

	#test data
	df_test = df[index:-1]
	df_target_test = df_target[index:-1]

	return df_train , df_test , df_target_train , df_target_test

#split data into train and test sets
def randomized_split_data( df , df_target , f_factor):

	df_train , df_test , df_target_train , df_target_test = train_test_split( df , df_target , test_size = 1 - f_factor , random_state=True )

	return df_train , df_test , df_target_train , df_target_test


#scale data. Fit only to training data
def scale_data( df_train , df_test  ):

	#scale data: Note that scaling is only fit on training data
	scaler = StandardScaler()
	scaler.fit(df_train)

	#scale train and test data
	df_train = pd.DataFrame( scaler.fit_transform(df_train) , columns = df_train.columns )
	df_test = pd.DataFrame( scaler.fit_transform(df_test) , columns = df_test.columns )	

	return df_train , df_test


#Perform SVD on train and test data. Keep N_trunc features
def SVD_truncation( df_train , df_test  , N_trunc ):

	
	pca = PCA( n_components = N_trunc )
	pca.fit( df_train )

	F_pca_train = pca.transform( df_train  )
	F_pca_test = pca.transform( df_test )


	singular_vals = pca.singular_values_
	var = pca.explained_variance_ratio_


	total_var = np.sum( singular_vals )

	# plt.title("1 - Explaned Variance Ratio")
	# plt.plot( 1 - singular_vals.cumsum()/total_var , '-o' )
	# plt.yscale('log')
	# plt.grid()
	# plt.show()

	cov = pca.get_covariance()
	
	#convert df_pca to pandas dataframe
	df_pca_train = pd.DataFrame()
	df_pca_train.index = df_train.index

	df_pca_test = pd.DataFrame()
	df_pca_test.index = df_test.index

	for n in range(N_trunc):

		pca_feature_name = "Feature_" + str(n) 

		df_pca_train[ pca_feature_name ] = F_pca_train[ : , n ]
		df_pca_test[ pca_feature_name ] = F_pca_test[ : , n ]

	
	# #plot how different features change with time
	# for n in range(N_trunc):
	# 	pca_feature_name = "Feature_" + str(n)
	# 	plt.plot(df_pca_train[ pca_feature_name ])

	# plt.grid()
	# plt.show()

	return df_pca_train , df_pca_test


#random forrest regression fit
def RFC_fit( df_train , df_target_train , target_shift , max_depth ):

	#split training data into fit and validate data
	f_factor = 0.7

	#Two options for validation: randomized or sequential
	#df_fit , df_validate , df_target_fit , df_target_validate = split_data( df_train , df_target_train , f_factor )
	df_fit , df_validate , df_target_fit , df_target_validate = randomized_split_data( df_train , df_target_train , f_factor )

	
	###majority and minority data
	N_maj = df_fit[ df_target_fit.values==0 ].shape[0]
	N_min = df_fit[ df_target_fit.values==1 ].shape[0]

	### rebalance training set if needed
	#df_oversampled, df_oversampled_target = resample( df_fit[ df_target_fit.values == 1 ] , df_target_fit[ df_target_fit.values == 1 ], replace=True ,  n_samples= N_maj - N_min , random_state=True)
	#df_fit = df_fit.append( df_oversampled )
	#df_target_fit = df_target_fit.append( df_oversampled_target )

	#print( df_fit[ df_target_fit.values==0 ].shape[0] )
	#print( df_fit[ df_target_fit.values==1 ].shape[0] )
	
	#print("Balanced Fit Data:" , df_target_fit.sum()/len(df_target_fit) == 0.5 )
	

	filename = "./data/T_"+str(target_shift)+"_D_"+str(max_depth)+"_RFC.joblib"

	try:

		### raise Exception
		clf = joblib.load(filename)

		train_score = clf.score( df_fit , df_target_fit  )
		test_score = clf.score( df_validate , df_target_validate  )		
		
	except Exception as e:

		#optimize over other paramters
		#K-fold cross validate?
		depth_pnts = max_depth
		depth_space = np.linspace( 1 , max_depth , depth_pnts )

		test_scores = np.zeros( depth_pnts )
		train_scores = np.zeros( depth_pnts )

		cnt = 0
		for depth in depth_space:

			clf = RandomForestClassifier( max_depth=depth, random_state=True )
			clf.fit( df_fit , df_target_fit )

			train_score = clf.score( df_fit , df_target_fit  )
			test_score = clf.score( df_validate , df_target_validate  )

			train_scores[cnt] = train_score
			test_scores[cnt] = test_score

			cnt = cnt + 1

		#find best test score
		index = np.argmax( test_scores  )
		optimal_depth = depth_space[ index ]

		clf = RandomForestClassifier( max_depth=optimal_depth , random_state=True )
		clf.fit( df_fit , df_target_fit )
		RF_path = clf.decision_path(df_fit)

		train_score = clf.score( df_fit , df_target_fit  )
		test_score = clf.score( df_validate , df_target_validate  )

		print()
		print("Optimal Random Forest Fit")
		print("Fit Time Frame:" ,  df_train.index[0] , 'until' , df_train.index[-1] )
		print("Target Return Shift:", str(target_shift) +' days')
		print("Number of Features:" , len(df_train.columns) )
		print("Total Number of Training Datapoints:", len(df_train) )
		print("Optimal Random Forest Regression Train Score:", train_score)
		print("Optimal Random Forest Regression Validation Score:", test_score)
		print("Optimal Random Forest depth:", optimal_depth )


		#save model
		joblib.dump( clf , filename )


	return clf


#plot confusion matrix
def plot_confusion_matrix( df_target , r_pred , title_str  ):

	conf_mat = confusion_matrix( r_pred , df_target )

	plt.title(title_str + " Confusion Matrix")
	plt.imshow(conf_mat , extent=[-2,2,2,-2] )
	plt.ylabel("Predicted Label")
	plt.xlabel("Actual Label")
	plt.xticks([-1,1])
	plt.yticks([-1,1])
	plt.colorbar()
	plt.show()

	print( "Confusion Matrix:" , conf_mat )


#plot ROC curves
def plot_ROC( df_target , r_pred , title_str  ):

	# compute ROC, larger return is positive_label = 1
	fpr, tpr, thresholds = metrics.roc_curve( df_target , r_pred , pos_label=1 )
	

	plt.title( title_str + " ROC Curve" )
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")

	g_pnts = 50
	g_space = np.linspace( 0 , 1, g_pnts )

	plt.plot(g_space,g_space , '--', label = 'Totally Random Classifier')
	plt.plot( fpr, tpr, label = 'RFC Classifier')
	plt.xlim(0,1)
	plt.ylim(0,1.03)
	plt.legend()
	plt.grid()
	plt.show()



