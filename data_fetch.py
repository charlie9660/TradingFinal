#import sql packages
import credentials
import sqlconnector as sql
import pandas as pd
import numpy as np
from functools import reduce

#function to check if element is number, used to remove Nans in dataframe
def isnumber(x):
	try:
		float(x)
		return True
	except:
		return False

#load data for scale sizes
def load_scale():

	file = "./data/scale.pkl"

	try:
		
		### raise Exception
		df = pd.read_pickle( file ) 
		
	except Exception as e:

		print("Fetching Data...")

		query = ("SELECT fund_id, AVG(NULLIF(total_assets, 0)) as total_asset"
		" FROM fund_balance" 
		" GROUP BY fund_id")
		
		#sql connector object
		s = sql.sqlconnector(credentials.HOST,credentials.USER,credentials.PASSWORD,credentials.PORT,'product_mutual')
		s.connect()

		df = s.fetch( query )
		df['total_asset'] = pd.to_numeric(df['total_asset'])
		df = df[ df.total_asset > 5e9 ]

		#save data to file
		df.to_pickle(file)  

	return df

#load data for given query
def load_table(  table  ):

	standard_query  = ( "select * from {table} where statistic_date > '2017-01-01'"
	"and `benchmark` = 1 and fund_id in"
	" (select fund_id from fund_info where foundation_date < '2018-01-01' and is_main_fund = 1 "
	"and fund_type = '股票型基金')" )

	query =  standard_query.format( table = table ) 

	file = "./data/T_"+str(table)+"_data.pkl"

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

#load data for nv table
def load_nv_table(  query  ):

	file = "./data/fund_nv.pkl"

	try:
		
		raise Exception
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

#load all data from file, return target varible as pandas dataframe
def load_feature_data( info_shift  , target_shift   ):

	fund_scale_selected = load_scale()

	# fund_return  = 'fund_monthly_return'
	# fund_risk = 'fund_monthly_risk'
	# fund_risk2 = 'fund_monthly_risk2'
	# fund_subsidiary = 'fund_monthly_subsidiary'
	# fund_subsidiary2 = 'fund_monthly_subsidiary2'
	# fund_subsidiary3 = 'fund_monthly_subsidiary3'
	
	fund_return  = 'fund_weekly_return'
	fund_risk = 'fund_weekly_risk'
	fund_risk2 = 'fund_weekly_risk2'
	fund_subsidiary = 'fund_weekly_subsidiary'
	fund_subsidiary2 = 'fund_weekly_subsidiary2'
	fund_subsidiary3 = 'fund_weekly_subsidiary3'
	
	# fund_return  = 'fund_daily_return'
	# fund_risk = 'fund_daily_risk'
	# fund_risk2 = 'fund_daily_risk2'
	# fund_subsidiary = 'fund_daily_subsidiary'
	# fund_subsidiary2 = 'fund_daily_subsidiary2'
	# fund_subsidiary3 = 'fund_daily_subsidiary3'
	
	filename = "./data/T_"+str(target_shift)+"_I_"+str(info_shift)+"_data.pkl"
	filename_target = "./data/T_"+str(target_shift)+"_I_"+str(info_shift)+"_target_data.pkl"

	try:

	
		### raise Exception
		df = pd.read_pickle( filename )
		print("Feature Dataset Loaded from file")


	except Exception as e1:

		#load all datasets
		df_return = load_table( fund_return  )
		df_risk = load_table( fund_risk )
		df_risk2 = load_table( fund_risk2 )
		df_subsidiary = load_table( fund_subsidiary )
		df_subsidiary2 = load_table( fund_subsidiary2 )
		df_subsidiary3 = load_table( fund_subsidiary3 )
					
		#feature data
		df = [ df_return , df_risk , df_risk2 , df_subsidiary , df_subsidiary2 , df_subsidiary3 ]
		
		#remove from memeory
		del df_return
		del df_risk
		del df_risk2
		del df_subsidiary
		del df_subsidiary2
		del df_subsidiary3

		df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['statistic_date','fund_id'], how='outer',suffixes=('', '_drop')), df )
		df_merged.drop([col for col in df_merged.columns if 'drop' in col], axis=1, inplace=True)
		df_merged = df_merged.fillna(value=np.nan)

		df_merged = df_merged.loc[df_merged['fund_id'].isin(fund_scale_selected.fund_id)]
		df_merged.statistic_date = pd.to_datetime(df_merged.statistic_date)

		drop_cols = ['entry_time' , 'update_time' , 'benchmark' , 'fund_name']
		df = df_merged.drop(columns=drop_cols)
		df = df.pivot(index='statistic_date', columns='fund_id'  , values = None )
		df.columns = ['_'.join(col).strip() for col in df.columns.values]

		df.to_pickle(filename)  
		

	#fill forward nans
	df = df.ffill()
	df = df.dropna( axis='columns' )

	return df 

#load all data from file, return target varible is returns for fund_id as pandas dataframe
def load_all_tables( fund_id , info_shift  , target_shift   ):

	#date range
	start_date = pd.Timestamp( '2018-01-01' )
	end_date = pd.Timestamp( '2021-05-01' )

	df = load_feature_data( info_shift , target_shift )

	fund_scale_selected = load_scale()

	filename = "./data/edit_T_"+str(target_shift)+"_I_"+str(info_shift)+"_F_"+str(fund_id)+"_data.pkl"
	filename_target = "./data/T_"+str(target_shift)+"_I_"+str(info_shift)+"_F_"+str(fund_id)+"_target_data.pkl"

	try:

	
		### raise Exception
		df = pd.read_pickle( filename )
		df_target = pd.read_pickle( filename_target ) 


	except Exception as e1:

		
		fund_nv =  """SELECT * from 
		fund_nv
		WHERE
		statistic_date > '2018-01-01'
		AND fund_id = {fund_id}
		"""

		fund_nv =  fund_nv.format( fund_id = fund_id )
		
		#load nv data
		df_nv = load_nv_table( fund_nv  )

		#drop irre columns
		drop_cols = ['entry_time' , 'update_time' , 'data_source' , 'fund_name' , 'fund_id']
		df_nv = df_nv.drop(columns=drop_cols)

		#order by statistic date and pad daily
		df_nv.index = pd.to_datetime(df_nv.statistic_date)
		df_nv = df_nv.sort_index()
		df_nv = df_nv.resample('D').ffill()

		#compute 'target shift ~ 1 month' look ahead return
		df_nv['target_return'] =  df_nv[ 'added_nav' ].shift( periods = -target_shift , freq=None ) / df_nv[ 'added_nav'] 
		df_nv = df_nv[ 0 : -target_shift ]

		#pad time to daily for full dataset
		df = df.resample('D').ffill()


		#make sure dates agree
		start_date = max( start_date ,   df.index[0] , df_nv.index[0]   )
		end_date = min( end_date ,  df.index[-1] , df_nv.index[-1]   )

		#reindex so that dates agree
		df = df.loc[ (df.index > start_date ) & (df.index <= end_date ) ]
		df_nv = df_nv.loc[  (df_nv.index > start_date ) & (df_nv.index <= end_date )  ]
		df_nv = df_nv.loc[ df.index  ]

		#the target varible
		df_target = df_nv['target_return']
		
		#assume that data takes some time to arrive, info_shift is order 1 week 
		df = df.shift( periods = info_shift , freq=None )
		df = df.iloc[info_shift::]
		df_target = df_target.iloc[info_shift::]


		#save target data to file
		df_target = pd.DataFrame(data=df_target, index=df.index)
		df_target.to_pickle(filename_target)  
		df.to_pickle(filename)


	return df,  df_target

