import pymongo as pm

class mongodb:
    def __init__(self, hostname='localhost', port=27017):
        self.conn = pm.MongoClient(hostname, port)
    
    def set_db(self, dbname):
        self.db = self.conn.get_database(dbname)
        return self.db

    def set_collection(self, collection_name):
        self.collection = self.db.get_collection(collection_name)
        return self.collection
    
    def get_collection(self):
        try:
            return self.collection
        except:
            print("you need to use set_collection before use get_collection")
            return None