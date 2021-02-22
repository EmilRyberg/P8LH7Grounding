import sqlite3
import numpy as np

# This class works as a handler for the database. It has functions that should be called from outside
# but also has a few functions like the 'create_table' that should be modified from here and called by running this module.
class DatabaseHandler:
    def __init__(self):
        self.conn = sqlite3.connect('grounding.db')
        print("Opened database successfully")

    def __del__(self):
        self.conn.close()

    def create_table(self): # This function has to be manually modified and called from this file if we wish to make a new table
        self.conn.execute('''CREATE TABLE FEATURES
                                (KEY INTEGER PRIMARY KEY AUTOINCREMENT,
                                NAME TEXT, 
                                FEATURE_VECTOR TEXT);''')
        print("Table created successfully")

    def insert_feature(self, name, feature_vector):
        self.conn.execute("INSERT INTO FEATURES (NAME, FEATURE_VECTOR) \
                                   VALUES (?, ?);", (name, ",".join([str(x) for x in feature_vector])))
        self.conn.commit()
        print("Records created successfully")

    def select(self, name):
        result = self.conn.execute("SELECT FEATURE_VECTOR from FEATURES \
                        WHERE NAME=?;", (name, ))
        features = None
        for row in result:
            features=row[0]
        print("Select operation done successfully")
        if features is not None:
            features = np.fromstring(features, dtype=float, sep = ', ')
        return features

    def update(self, name, feature_vector):
        self.conn.execute("UPDATE FEATURES set FEATURE_VECTOR = ? where NAME = ?;", (str(feature_vector), name))
        self.conn.commit()
        print("Update operation was successful")

    def delete(self, name):
        self.conn.execute("DELETE from FEATURES where NAME = ?;", (name,))
        self.conn.commit()
        print("Total number of rows deleted :", self.conn.total_changes)


if __name__ == "__main__":
    db = DatabaseHandler()
    feature = np.random.rand(5)
    #db.conn.execute("DROP TABLE FEATURES")
    #db.create_table()
    #db.delete("black cover")
    db.insert_feature("blue cover", feature)
    #features = db.select("red cover")
    db.conn.close()
