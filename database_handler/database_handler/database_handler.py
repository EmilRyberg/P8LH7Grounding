import sqlite3
import numpy as np
import jsonpickle
from ner_lib.command_builder import Task, TaskType, SpatialType, ObjectEntity, SpatialDescription

# This class works as a handler for the database. It has functions that should be called from outside
# but also has a few functions like the 'create_table' that should be modified from here and called by running this module.
class DatabaseHandler:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        #print("Opened database successfully")

    def __del__(self):
        self.conn.close()

    def create_table(self): # This function has to be manually modified and called from this file if we wish to make a new table
        self.conn.execute('''CREATE TABLE FEATURES
                                (KEY INTEGER PRIMARY KEY AUTOINCREMENT,
                                NAME TEXT, 
                                FEATURE_VECTOR TEXT);''')

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

    def get_task(self, word):
        result = self.conn.execute("SELECT TASK_INFO.TASK_ID, TASK_NAME from TASK_WORDS inner join TASK_INFO on "
                                   "TASK_WORDS.TASK_ID=TASK_INFO.TASK_ID where WORD=?;", (word, ))
        task_id = None
        task_name = None
        for row in result:
            task_id = row[0]
            task_name = row[1]
        return (task_id, task_name)

    def get_task_name(self, task_id):
        result = self.conn.execute("SELECT TASK_NAME from TASK_INFO where TASK_ID=?;", (task_id,))
        task_name = None
        for row in result:
            task_name = row[0]
        return task_name

    def get_task_id(self, task_name):
        result = self.conn.execute("SELECT TASK_ID from TASK_INFO where TASK_NAME=?;", (task_name,))
        task_id = None
        for row in result:
            task_id = row[0]
        return task_id

    def get_sub_tasks(self, task_id):
        result = self.conn.execute("SELECT SUB_TASK_ID, TASK_NAME, JSON_TASK from "
                                   "TASK_SUBTASKS inner join TASK_INFO on TASK_SUBTASKS.SUB_TASK_ID=TASK_INFO.TASK_ID "
                                   "where "
                                   "TASK_SUBTASKS.TASK_ID=? order by TASK_ORDER;", (task_id,))
        sub_task_ids = []
        sub_task_names = []
        task_objects = []
        for row in result:
            sub_task_ids.append(row[0])
            sub_task_names.append(row[1])
            task_json = row[2]
            task_object = None
            if task_json is not None:
                task_object = jsonpickle.decode(row[2])
            task_objects.append(task_object)
        sub_tasks = [sub_task_ids, sub_task_names, task_objects]
        return sub_tasks

    def add_sub_task(self, task_id, sub_task_id, task=None):
        current_sub_tasks = self.get_sub_tasks(task_id)
        current_amount_of_sub_tasks = len(current_sub_tasks[0])
        json_task = None
        if task:
            json_task = jsonpickle.encode(task)
        self.conn.execute("INSERT INTO TASK_SUBTASKS (TASK_ID, TASK_ORDER, SUB_TASK_ID, "
                          "JSON_TASK) VALUES (?, ?, ?, ?);", (task_id, current_amount_of_sub_tasks+1,
                                                                         sub_task_id, json_task))
        self.conn.commit()

    def add_task(self, task_name, words):
        self.conn.execute("INSERT INTO TASK_INFO (TASK_NAME) VALUES (?);", (task_name, ))
        self.conn.commit()
        task_id = self.get_task_id(task_name)
        already_known = []
        for word in words:
            word_exists = self.conn.execute("SELECT * FROM TASK_WORDS WHERE WORD=?;", (word, ))
            if word_exists.fetchone():
                already_known.append(word)
            else:
                self.add_word_to_task(task_id, word)
        return task_id, already_known

    def object_exists(self, object_name):
        already_known = False
        object_exists = self.conn.execute("SELECT * FROM FEATURES WHERE NAME=?;", (object_name, ))
        if object_exists.fetchone():
            already_known = True
        return already_known

    def add_new_object(self, object_name, feature_vector):
        already_known = False
        object_exists = self.conn.execute("SELECT * FROM FEATURES WHERE NAME=?;", (object_name, ))
        if object_exists.fetchone():
            already_known = True
        if not already_known:
            self.conn.execute("INSERT INTO FEATURES (NAME,FEATURE_VECTOR) VALUES (?,?);", (object_name, feature_vector))
            self.conn.commit()
        object_id = self.get_feature(object_name)
        return object_id, already_known

    def add_word_to_task(self, task_id, word):
        try:
            self.conn.execute("INSERT INTO TASK_WORDS (TASK_ID, WORD) VALUES (?,?);", (task_id, word))
            self.conn.commit()
        except:
            raise Exception("Unable to add word:", word, " to task: ", task_id)

    def update(self, name, feature_vector):
        self.conn.execute("UPDATE FEATURES set FEATURE_VECTOR = ? where NAME = ?;", (",".join([str(x) for x in feature_vector]),name))
        self.conn.commit()

    def delete(self, name):
        self.conn.execute("DELETE from FEATURES where NAME = ?;", (name,))
        self.conn.commit()

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
        ''', (name,))
        data = result.fetchall()
        if data is None or len(list(data)) == 0:
            return None, None, None
        row = list(data)[0]
        return row[1], row[2], row[3]


if __name__ == "__main__":
    db = DatabaseHandler("../../dialog_flow/nodes/grounding.db")
    already_known = db.add_task("Move black", ["get", "put"])
    # task = Task("pick up")
    # object_entity = ObjectEntity("blue cover")
    # spatial_description = SpatialDescription(spatial_type=SpatialType.NEXT_TO)
    # spatial_description.object_entity.name = "black cover"
    # object_entity.spatial_descriptions.append(spatial_description)
    # task.objects_to_execute_on.append(object_entity)
    # db.add_sub_task(5, 1, task)
    # tasks = db.get_sub_tasks(5)
    print(already_known)
    #test = db.get_sub_tasks(6)
    #db.conn.close()
