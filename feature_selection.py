#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import credentials
import sqlconnector as sql

#import linear models and test_train_split_funtion
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso ,  LinearRegression

#function to check if element is number
def isnumber(x):
	try:
		float(x)
		return True
	except:
		return False

#load data for given query
def load_data(  query  , label ):

	file = "./data/"+str(label)+"temp.pkl"

	try:
		
		### raise Exception
		df = pd.read_pickle( file ) 
		
	except Exception as e:

		print("Fetching Data...")
		
		#sql connector object
		s = sql.sqlconnector(credentials.HOST,credentials.USER,credentials.PASSWORD,credentials.PORT,'product_mutual')
		s.connect()

		df = s.fetch( query )

		#save data to file
		df.to_pickle(file)  

	return df

#load all data from file
def load_all( fund_id , start_date , end_date ):

	fund_weekly_return = """SELECT * from 
	fund_daily_return
	where 
	statistic_date > '2015-01-01'
	AND 
	statistic_date < '2021-01-01'
	AND
	benchmark = 1
	AND
	fund_id = 3
	LIMIT 10000"""

	fund_weekly_risk = """SELECT * from 
	fund_daily_risk
	where 
	statistic_date > '2015-01-01'
	AND 
	statistic_date < '2021-01-01'
	AND
	benchmark = 1
	AND
	fund_id = 3
	LIMIT 10000"""

	fund_weekly_risk2 = """SELECT * from 
	fund_daily_risk2
	where 
	statistic_date > '2015-01-01'
	AND 
	statistic_date < '2021-01-01'
	AND
	benchmark = 1
	AND
	fund_id = 3
	LIMIT 10000"""

	fund_weekly_subsidiary = """SELECT * from 
	fund_daily_subsidiary
	where 
	statistic_date > '2015-01-01'
	AND 
	statistic_date < '2021-01-01'
	AND
	benchmark = 1
	AND
	fund_id = 3
	LIMIT 10000"""

	fund_weekly_subsidiary2 = """SELECT * from 
	fund_daily_subsidiary2
	where 
	statistic_date > '2015-01-01'
	AND 
	statistic_date < '2021-01-01'
	AND
	benchmark = 1
	AND
	fund_id = 3
	LIMIT 10000"""


	fund_weekly_subsidiary3 = """SELECT * from 
	fund_daily_subsidiary3
	where 
	statistic_date > '2015-01-01'
	AND 
	statistic_date < '2021-01-01'
	AND
	benchmark = 1
	AND
	fund_id = 3
	LIMIT 10000"""

	#load all datasets
	df_return = load_data( fund_weekly_return , 0 )
	df_risk = load_data( fund_weekly_risk ,  1 )
	df_risk2 = load_data( fund_weekly_risk2 , 2 )
	df_subsidiary = load_data( fund_weekly_subsidiary , 3 )
	df_subsidiary2 = load_data( fund_weekly_subsidiary2 , 4 )
	df_subsidiary3 = load_data( fund_weekly_subsidiary3 , 5 )

	#reindex by date
	df_return = df_return.set_index( 'statistic_date' )
	df_risk = df_risk.set_index( 'statistic_date' )
	df_risk2 = df_risk2.set_index( 'statistic_date' )
	df_subsidiary = df_subsidiary.set_index( 'statistic_date' )
	df_subsidiary2 = df_subsidiary2.set_index( 'statistic_date' )
	df_subsidiary3 = df_subsidiary3.set_index( 'statistic_date' )

	return df_return , df_risk  , df_risk2  , df_subsidiary  , df_subsidiary2  , df_subsidiary3 

#get all tables
df_return , df_risk  , df_risk2  , df_subsidiary  , df_subsidiary2  , df_subsidiary3  = load_all( fund_id = 1 , start_date = '2015-01-01' , end_date = '2021-01-01' )
df = pd.concat( [ df_risk , df_risk2 , df_subsidiary , df_subsidiary2 , df_subsidiary3  ] , axis = 1 )
df.index = pd.to_datetime(df.index)
df = df.resample('1d').pad()


df_return.index = pd.to_datetime(df_return.index)
df_return = df_return.resample('1d').pad()

#cut dates to agree
start_date = max( df_return.index[0] , df.index[0] )
end_date = min( df_return.index[-1] , df.index[-1] )

df = df.loc[ (df.index > start_date ) & (df.index <= end_date ) ]
df_return = df_return.loc[ (df_return.index > start_date ) & (df_return.index <= end_date ) ]


#drop non-numberic columns in df
drop_cols = [ 'fund_id' , 'benchmark' , 'fund_name' , 'entry_time' , 'update_time' ]
df = df.drop( columns = drop_cols )

#shift returns backward one month, i.e. assuming that get returns after shift days lag
shift = 30
df = df.shift(periods=shift )
df = df.iloc[shift:, :]

#target varible
df_return = df_return.iloc[shift:,:]
df_return = df_return.fillna(method="ffill")
df_target = df_return['month_return']

#drop invalid columns
df = df[ df.applymap( isnumber  )  ]
df = df.fillna(method="ffill")
df = df.dropna(  axis = 'columns' )

#number of features
p = len( df.columns )

#number of datapoints
N = len( df )

### split test and train data
df_train , df_test , df_target_train , df_target_test = train_test_split( df , df_target , test_size=0.25 , random_state=True)

### fit linear model
reg = LinearRegression()
reg.fit( df_train , df_target_train  )
train_score = reg.score( df_train , df_target_train  )
test_score = reg.score( df_test , df_target_test  )

print("Linear Regression Train Score:" , train_score)
print("Linear Regression Test Score:" , test_score)
print("Number of Features:" , p , ',' , "Number of Datapoints:" , N )


### ridge regression parameters
tol = 1e-3
max_iter = 1e+5

alpha_pnts = 10 
alpha_min = 1e-3
alpha_max = 10.0
alpha_space = np.logspace( alpha_min , alpha_max , alpha_pnts )

coef_reg = np.zeros( [ alpha_pnts , p ] )
test_scores = np.zeros( alpha_pnts )
train_scores = np.zeros( alpha_pnts )

cnt = 0
for alpha in alpha_space:

	#ridge regress
	clf = Ridge( alpha=alpha , tol=tol , max_iter=max_iter )
	clf.fit( df_train , df_target_train )
	train_score = clf.score( df_train , df_target_train  )
	test_score = clf.score( df_test , df_target_test  )

	coef = clf.coef_
	coef_reg[ cnt , : ] = coef 
	
	train_scores[cnt] = train_score
	test_scores[cnt] = test_score
	cnt = cnt + 1
	
#find best regularization parameter
index = np.argmax( test_scores  )
alpha_max = alpha_space[ index ]

train_score = train_scores[index]
test_score = test_scores[index]


print("")
print("")
print("Optimal Lasso Regression Alpha:" , alpha_max )
print("Optimal Lasso Regression Train Score:", train_score )
print("Optimal Lasso Regression Test Score:", test_score )
print("Number of Features:" , p , ',' , "Number of Datapoints:" , N )

plt.title("Test and Train Loss as Funtion of Regularization Parameter alpha ")
plt.plot( alpha_space , train_scores , '-o' , label = "Train Score" )
plt.plot( alpha_space ,  test_scores , '-o' , label = "Test Score" )
plt.plot( alpha_max , train_score , 'X' ,label='Optimal Train Loss' , color = 'k' , markersize = 16 )
plt.plot( alpha_max , test_score , '*' ,label='Optimal Test Loss' , color = 'k' , markersize = 19 )
plt.xlabel('Regularization Parameter alpha')
plt.ylabel('Test and Train Score')
plt.xscale('log')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


#need to return parameters
coef = clf.coef_
intercept = clf.intercept_
feature_names = clf.feature_names_in_

#number of significant features to plot
N_sig = 10

#get 10 largest coeficnets
pos_coef = np.abs( coef )

sorted_index_array = np.argsort( pos_coef )
sorted_coef = coef[sorted_index_array]

sig_coef = sorted_coef[::-1][ 0:N_sig ]
sig_features = feature_names[  sorted_index_array ][::-1][ 0:N_sig ]

print("")
print("Most Significant Features:")
print("Feature Weight , Feature Name")
for i in range(N_sig):
	print( sig_coef[i] , sig_features[i] )

for i in sorted_index_array[::-1][0:N_sig]:
	plt.plot( alpha_space , coef_reg[:,i] , '-.' , label = feature_names[i] )

plt.title( str(N_sig)+" Largest Ridge Regression Coefficient Values \n as Function of Regularization Parameter alpha")
plt.xlabel('Regularization Parameter alpha')
plt.ylabel('Ridge Regression Coefficient Values')
plt.xscale('log')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()




#### to do
### 1. Feature selection
### 2. Random Forrest
### 3. Do PCA on data, then compute?
