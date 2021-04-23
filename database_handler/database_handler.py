import sqlite3
import numpy as np

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
        result = self.conn.execute("SELECT TASK_ID from TASK_WORDS where WORD=?;", (word, ))
        task_id = None
        for row in result:
            task_id = row[0]
        task_name = self.get_task_name(task_id)
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

    def get_sub_tasks(self, task_name):
        result = self.conn.execute("SELECT TASK_SUBTASKS from TASK_INFO where TASK_NAME=?;", (task_name,))
        sub_tasks = None
        for row in result:
            sub_tasks = row[0]
        return sub_tasks

    def add_sub_task(self, task_name, sub_task_id):
        current_sub_tasks = self.get_sub_tasks(task_name)
        if current_sub_tasks is None:
            sub_tasks = str(sub_task_id)+","
        else:
            sub_tasks = current_sub_tasks + str(sub_task_id) + ","
        self.conn.execute("UPDATE TASK_INFO set TASK_SUBTASKS = ? WHERE TASK_NAME = ?;", (sub_tasks, task_name))
        self.conn.commit()

    def add_task(self, task_name, words):
        try:
            self.conn.execute("INSERT INTO TASK_INFO (TASK_NAME) VALUES (?);", (task_name, ))
            self.conn.commit()
            task_id = self.get_task_id(task_name)
            for word in words:
                self.add_word_to_task(task_id, word)
        except:
            raise Exception("Unable to add task:", task_name)


    def add_word_to_task(self, task_id, word):
        try:
            self.conn.execute("INSERT INTO TASK_WORDS (TASK_ID, WORD) VALUES (?,?);", (task_id, word))
            self.conn.commit()
        except:
            raise Exception("Unable to add word:", word, " to task: ",task_id)

    def update(self, name, feature_vector):
        self.conn.execute("UPDATE FEATURES set FEATURE_VECTOR = ? where NAME = ?;", (",".join([str(x) for x in feature_vector]),name))
        self.conn.commit()

    def delete(self, name):
        self.conn.execute("DELETE from FEATURES where NAME = ?;", (name,))
        self.conn.commit()


if __name__ == "__main__":
    db = DatabaseHandler("../dialog_flow/nodes/grounding.db")
    db.conn.close()
