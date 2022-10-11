from dotenv import load_dotenv
from .mongoDBOps import MongoDBManagement
import os
load_dotenv()
MONGO_CONN_STR = os.getenv("MONGO_CONN_STR")


def save_logs_in_mongo(collection_name:str, record):
    mongo = MongoDBManagement(MONGO_CONN_STR)
    mongo.getDatabase(db_name='onespace')
    mongo.insertRecord(db_name='onespace', collection_name = collection_name, record = record)
    print("\n")
    print("=====" * 13)
    print('Experiment logs have been saved in MongoDB successfully')
    print("=====" * 13)

