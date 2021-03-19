import sqlite3
import numpy as np

# This class works as a handler for the database. It has functions that should be called from outside
# but also has a few functions like the 'create_table' that should be modified from here and called by running this module.
class DatabaseHandler:
    def __init__(self):
        self.conn = sqlite3.connect('../grounding.db')
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

    def get_feature(self, name):
        result = self.conn.execute("SELECT FEATURE_VECTOR from FEATURES \
                        WHERE NAME=?;", (name, ))
        features = None
        for row in result:
            features=row[0]
        print("Select operation done successfully")
        if features is not None:
            features = np.fromstring(features, dtype=float, sep = ', ')
        return features

    def get_all_features(self):
        result = self.conn.execute("SELECT NAME, FEATURE_VECTOR from FEATURES;")
        db_objects = []
        for row in result:
            name = row[0]
            features = row[1]
            db_objects.append((name, features))
        return db_objects

    def update(self, name, feature_vector):
        self.conn.execute("UPDATE FEATURES set FEATURE_VECTOR = ? where NAME = ?;", (",".join([str(x) for x in feature_vector]),name))
        self.conn.commit()
        print("Update operation was successful")

    def delete(self, name):
        self.conn.execute("DELETE from FEATURES where NAME = ?;", (name,))
        self.conn.commit()
        print("Total number of rows deleted :", self.conn.total_changes)


if __name__ == "__main__":
    db = DatabaseHandler()
    feature = np.array([0.71637168, 0.54895908, 0.69006481, 0.00490511, 0.22038214])
    # db.conn.execute("DROP TABLE FEATURES")
    # db.create_table()
    db.delete("black cover")
    # db.delete("blue cover")
    db.insert_feature("black cover", feature)
    #db.insert_feature("red cover", feature)
    #db.insert_feature("bottom cover", feature)
    #db.insert_feature("fuse", feature)
    # print(db.get_feature("green cover"))
    objects = db.get_all_features()
    print(objects)
    db.conn.close()
