from mysql.connector import connect, Error
import pandas as pd

class sqlconnector:
   
    def __init__(self,host,user,password,port,database):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.database = database
        self.connection = None
        
    def connect(self):
        try:
            self.connection = connect(
                host= self.host,
                user= self.user,
                password= self.password,
                port = self.port,
                database = self.database
            )
            print('Connection Success')
        except Error as e:
            print(e)
            print('Connection Failure')
            

    def fetch(self,query):
        """fetch data through SQL query

        :param query: sql query
        :param database: database to select from
        :return: a dataframe
        """  
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            num_fields = len(cursor.description)
            field_names = [i[0] for i in cursor.description]
            df = pd.DataFrame(result,columns = field_names)
            cursor.close()
            return df
        except Error as e:
            print(e)
            print('Retrying Connection...')
            self.connect()
            return self.fetch(query)
        return None