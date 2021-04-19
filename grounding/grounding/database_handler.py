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

    def create_static_locations_table(self):
        self.conn.execute('''
            CREATE TABLE STATIC_LOCATIONS (ID INTEGER PRIMARY KEY AUTOINCREMENT,
            X REAL,
            Y REAL,
            Z REAL);
        ''')

    def create_static_location_relations_table(self):
        self.conn.execute('''
            CREATE TABLE STATIC_LOCATION_RELATIONS (ID INTEGER PRIMARY KEY AUTOINCREMENT,
            NAME TEXT,
            STATIC_LOCATION_ID INTEGER,
            FOREIGN KEY(STATIC_LOCATION_ID) REFERENCES STATIC_LOCATIONS(ID));
        ''')

    def get_location_by_name(self, name):
        result = self.conn.execute('''
            SELECT SLR.NAME, SL.X, SL.Y, SL.Z FROM STATIC_LOCATION_RELATIONS AS SLR INNER JOIN STATIC_LOCATIONS AS SL ON SL.ID = SLR.STATIC_LOCATION_ID WHERE SLR.NAME = ?;
        ''', (name, ))
        if result is None or len(list(result)) == 0:
            return None
        row = list(result)[0]
        return row[1], row[2], row[3]

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
    db = DatabaseHandler("../../dialog_flow/grounding.db")
    x, y, z = db.get_location_by_name("test")
    print(f"x: {x}, y: {y}, z: {z}")
    #feature = np.array([1, 1, 1, 1, 1])
    # db.conn.execute("DROP TABLE FEATURES")
    # db.create_table()
    #db.delete("black cover")
    #db.insert_feature("black cover", feature)
    #db.delete("green cover")
    #db.insert_feature("red cover", feature)
    #db.insert_feature("bottom cover", feature)
    #db.insert_feature("fuse", feature)
    # print(db.get_feature("green cover"))
    #objects = db.get_all_features()
    #print(objects)
    db.conn.close()
