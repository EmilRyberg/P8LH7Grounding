import sqlite3
import numpy as np

# This class works as a handler for the database. It has functions that should be called from outside
# but also has a few functions like the 'create_table' that should be modified from here and called by running this module.
class DatabaseHandler:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        #print("Opened database successfully")

    def __del__(self):
        self.conn.close()

    def create_table(self): # This function has to be manually modified and called from this file if we wish to make a new table
        self.conn.execute('''CREATE TABLE FEATURES
                                (KEY INTEGER PRIMARY KEY AUTOINCREMENT,
                                NAME TEXT, 
                                FEATURE_VECTOR TEXT);''')

    def object_exists(self, object_name):
        already_known = False
        object_exists = self.conn.execute("SELECT * FROM FEATURES WHERE NAME=?;", (object_name, ))
        if object_exists.fetchone():
            already_known = True
        return already_known

    def insert_feature(self, name, feature_vector):
        self.conn.execute("INSERT INTO FEATURES (NAME, FEATURE_VECTOR) \
                                   VALUES (?, ?);", (name, ",".join([str(x) for x in feature_vector])))
        self.conn.commit()

    def get_feature(self, name):
        result = self.conn.execute("SELECT FEATURE_VECTOR from FEATURES \
                        WHERE NAME=?;", (name, ))
        features = None
        for row in result:
            features = row[0]
        if features is not None:
            features = np.fromstring(features, dtype=float, sep=',')
        return features

    def get_all_features(self):
        result = self.conn.execute("SELECT NAME, FEATURE_VECTOR from FEATURES;")
        db_objects = []
        for row in result:
            name = row[0]
            features = row[1]
            db_objects.append((name, np.fromstring(features, dtype=float, sep=',')))
        return db_objects

    def update(self, name, feature_vector):
        self.conn.execute("UPDATE FEATURES set FEATURE_VECTOR = ? where NAME = ?;", (",".join([str(x) for x in feature_vector]),name))
        self.conn.commit()

    def delete(self, name):
        self.conn.execute("DELETE from FEATURES where NAME = ?;", (name,))
        self.conn.commit()


if __name__ == "__main__":
    db = DatabaseHandler("grounding.db")
    feature = np.array([1, 1, 1, 1, 1])
    # db.conn.execute("DROP TABLE FEATURES")
    # db.create_table()
    db.delete("black cover")
    db.insert_feature("black cover", feature)
    db.delete("green cover")
    #db.insert_feature("red cover", feature)
    #db.insert_feature("bottom cover", feature)
    #db.insert_feature("fuse", feature)
    # print(db.get_feature("green cover"))
    objects = db.get_all_features()
    print(objects)
    db.conn.close()
