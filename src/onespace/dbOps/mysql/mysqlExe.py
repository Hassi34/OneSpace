from dotenv import load_dotenv
import os
from .mysqlDBOps import mysqlDBManagement
import pandas as pd

load_dotenv()
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASS = os.getenv('MYSQL_PASS')
MYSQL_DB = os.getenv('MYSQL_DB')

def save_logs_in_mysql(data, columns, project_name):
    mysql = mysqlDBManagement(host = MYSQL_HOST,
                            username = MYSQL_USER,
                            password = MYSQL_PASS,
                            database = MYSQL_DB)
    try:
        df = pd.DataFrame(dict(zip(columns, data)))
    except ValueError:
        df = pd.DataFrame(dict(zip(columns, data)), index = [columns[0]])
    except:
        print("!!! Sepecified Structures are not supported for MySQL")
    finally:
        pass
    mysql.saveDataFrameIntoDB(df, name = project_name)
    print("\n")
    print("=====" * 13)
    print('Experiment logs have been saved in MySQL successfully')
    print("=====" * 13)


