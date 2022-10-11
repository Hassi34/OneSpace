import pandas as pd
import pymongo

class MongoDBManagement:

    def __init__(self, conn_string):
        """
        This function sets the required url
        """
        try:
            if conn_string is not None:
                self.url = conn_string
        except Exception as e:
            raise Exception(f"(__init__): Something went wrong on initiation process\n" + str(e))

    def getMongoDBClientObject(self):
        """
        This function creates the client object for connection purpose
        """
        try:
            mongo_client = pymongo.MongoClient(self.url)
            return mongo_client
        except Exception as e:
            raise Exception("(getMongoDBClientObject): Something went wrong on creation of client object\n" + str(e))

    def closeMongoDBconnection(self, mongo_client):
        """
        This function closes the connection of client
        :return:
        """
        try:
            mongo_client.close()
        except Exception as e:
            raise Exception(f"Something went wrong on closing connection\n" + str(e))

    def isDatabasePresent(self, db_name):
        """
        This function checks if the database is present or not.
        :param db_name:
        :return:
        """
        try:
            mongo_client = self.getMongoDBClientObject()
            if db_name in mongo_client.list_database_names():
                return True
            else:
                return False
        except Exception as e:
            raise Exception("(isDatabasePresent): Failed on checking if the database is present or not \n" + str(e))

    def createDatabase(self, db_name):
        """
        This function creates database.
        :param db_name:
        :return:
        """
        try:
            database_check_status = self.isDatabasePresent(db_name=db_name)
            if not database_check_status:
                mongo_client = self.getMongoDBClientObject()
                database = mongo_client[db_name]
                return database
            else:
                mongo_client = self.getMongoDBClientObject()
                database = mongo_client[db_name]
                return database
        except Exception as e:
            raise Exception(f"(createDatabase): Failed on creating database\n" + str(e))

    def dropDatabase(self, db_name):
        """
        This function deletes the database from MongoDB
        :param db_name:
        :return:
        """
        try:
            mongo_client = self.getMongoDBClientObject()
            if db_name in mongo_client.list_database_names():
                mongo_client.drop_database(db_name)
                return True
        except Exception as e:
            raise Exception(f"(dropDatabase): Failed to delete database {db_name}\n" + str(e))

    def getDatabase(self, db_name):
        """
        This returns databases.
        """
        try:
            mongo_client = self.getMongoDBClientObject()
            db_name = mongo_client[db_name]
            return db_name
        except Exception as e:
            raise Exception(f"(getDatabase): Failed to get the database list")

    def getCollection(self, collection_name, db_name):
        """
        This returns collection.
        :return:
        """
        try:
            database = self.getDatabase(db_name)
            return database[collection_name]
        except Exception as e:
            raise Exception(f"(getCollection): Failed to get the database list.")

    def isCollectionPresent(self, collection_name, db_name):
        """
        This checks if collection is present or not.
        :param collection_name:
        :param db_name:
        :return:
        """
        try:
            
            database_status = self.isDatabasePresent(db_name=db_name)
            
            if database_status:
                
                database = self.getDatabase(db_name=db_name)
                
                if collection_name in database.list_collection_names():
                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
            raise Exception(f"(isCollectionPresent): Failed to check collection\n" + str(e))

    def createCollection(self, collection_name, db_name):
        """
        This function creates the collection in the database given.
        :param collection_name:
        :param db_name:
        :return:
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            if not collection_check_status:
                database = self.getDatabase(db_name=db_name)
                collection = database[collection_name]
                return collection
        except Exception as e:
            raise Exception(f"(createCollection): Failed to create collection {collection_name}\n" + str(e))

    def dropCollection(self, collection_name, db_name):
        """
        This function drops the collection
        :param collection_name:
        :param db_name:
        :return:
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
                collection.drop()
                return True
            else:
                return False
        except Exception as e:
            raise Exception(f"(dropCollection): Failed to drop collection {collection_name}")

    def insertRecord(self, db_name, collection_name, record):
        """
        This inserts a record.
        :param db_name:
        :param collection_name:
        :param record:
        :return:
        """
        try:
            # collection_check_status = self.isCollectionPresent(collection_name=collection_name,db_name=db_name)
            # print(collection_check_status)
            # if collection_check_status:
            collection = self.getCollection(collection_name=collection_name, db_name=db_name)
            collection.insert_one(record)
            return f"rows inserted "
        except Exception as e:
            raise Exception(f"(insertRecord): Something went wrong on inserting record\n" + str(e))

    def insertRecords(self, db_name, collection_name, records):
        """
        This inserts a record.
        :param db_name:
        :param collection_name:
        :param record:
        :return:
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name,db_name=db_name)
            # print(collection_check_status)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
            #record = list(records.values())
                collection.insert_many(records)
            return f"rows inserted "
        except Exception as e:
            raise Exception(f"(insertRecords): Something went wrong on inserting record\n" + str(e))

    def findfirstRecord(self, db_name, collection_name,query=None):
        """
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            print(collection_check_status)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
                print(collection)
                firstRecord = collection.find_one(query)
                return firstRecord
        except Exception as e:
            raise Exception(f"(findRecord): Failed to find record for the given collection and database\n" + str(e))

    def findAllRecords(self, db_name, collection_name, query=None):
        """
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
                if query is not None:
                    findAllRecords = collection.find(query,{"_id":0})
                else :
                    findAllRecords = collection.find({},{"_id":0})
                return findAllRecords
        except Exception as e:
            raise Exception(f"(findAllRecords): Failed to find record for the given collection and database\n" + str(e))

    def findRecordOnQuery(self, db_name, collection_name, query):
        """
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
                findRecords = collection.find(query)
                return findRecords
        except Exception as e:
            raise Exception(
                f"(findRecordOnQuery): Failed to find record for given query,collection or database\n" + str(e))

    def updateOneRecord(self, db_name, collection_name, query):
        """
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
                previous_records = self.findAllRecords(db_name=db_name, collection_name=collection_name)
                new_records = query
                updated_record = collection.update_one(previous_records, new_records)
                return updated_record
        except Exception as e:
            raise Exception(
                f"(updateRecord): Failed to update the records with given collection query or database name.\n" + str(
                    e))

    def updateMultipleRecord(self, db_name, collection_name, query):
        """
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
                previous_records = self.findAllRecords(db_name=db_name, collection_name=collection_name)
                new_records = query
                updated_records = collection.update_many(previous_records, new_records)
                return updated_records
        except Exception as e:
            raise Exception(
                f"(updateMultipleRecord): Failed to update the records with given collection query or database name.\n" + str(
                    e))

    def deleteRecord(self, db_name, collection_name, query):
        """
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
                collection.delete_one(query)
                return "1 row deleted"
        except Exception as e:
            raise Exception(
                f"(deleteRecord): Failed to update the records with given collection query or database name.\n" + str(
                    e))

    def deleteRecords(self, db_name, collection_name, query):
        """
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            if collection_check_status:
                collection = self.getCollection(collection_name=collection_name, db_name=db_name)
                collection.delete_many(query)
                return "Multiple rows deleted"
        except Exception as e:
            raise Exception(
                f"(deleteRecords): Failed to update the records with given collection query or database name.\n" + str(
                    e))

    def getDataFrameOfCollection(self, db_name, collection_name, query=None):
        """
        """
        try:
            all_Records = self.findAllRecords(collection_name=collection_name, db_name=db_name, query=query)
            dataframe = pd.DataFrame(all_Records)
            return dataframe
        except Exception as e:
            raise Exception(
                f"(getDataFrameOfCollection): Failed to get DatFrame from provided collection and database.\n" + str(e))

    def saveDataFrameIntoCollection(self, collection_name, db_name, dataframe):
        """
        """
        try:
            collection_check_status = self.isCollectionPresent(collection_name=collection_name, db_name=db_name)
            #dataframe_dict = json.loads(dataframe.T.to_json())
            dataframe_dict = dataframe.to_dict("records")
        
            if collection_check_status:
                self.insertRecords(collection_name=collection_name, db_name=db_name, records=dataframe_dict)
                return "Inserted"
            else:
                self.createDatabase(db_name=db_name)
                self.createCollection(collection_name=collection_name, db_name=db_name)
                self.insertRecords(db_name=db_name, collection_name=collection_name, records=dataframe_dict)
                return "Inserted"
        except Exception as e:
            raise Exception(
                f"(saveDataFrameIntoCollection): Failed to save dataframe value into collection.\n" + str(e))

    def getResultToDisplayOnBrowser(self, db_name, collection_name):
        """
        This function returns the final result to display on browser.
        """
        try:
            response = self.findAllRecords(db_name=db_name, collection_name=collection_name)
            result = [i for i in response]
            return result
        except Exception as e:
            raise Exception(
                f"(getResultToDisplayOnBrowser) - Something went wrong on getting result from database.\n" + str(e))
