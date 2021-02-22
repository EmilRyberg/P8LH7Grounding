import database_handler
import numpy as np
from typing import Optional

class Grounding:
    def __init__(self):
        self.db=database_handler.DatabaseHandler()

    def find_object(self, entity):
        found_object = False
        features_below_threshold = []
        distances = []

        db_features = self.db.select(entity)
        if db_features is None:
            affirmation = False
            # HRI.TextToSpeech("I dont know the object you asked about, do you want me to learn it?")
            # affirmation = NLP.AffirmationCheck
            if affirmation:
                self.learnNewObject(entity)
            else:
                # HRI.TextToSpeech("Okay.")
                return
        # features = vision.getBoundingBoxesWithFeatures()
        features = []

        for i, (bbox, feature) in enumerate(features):
            distance = self.embedding_distance(db_features, feature)
            is_below_threshold = self.is_same_object(db_features, feature, threshold=0.8) # TODO update threshold
            if is_below_threshold:
                found_object = True
                distances.append(distance)
                features_below_threshold.append(i)
        if found_object:
            best_match = self.find_best_match(features_below_threshold, distances)
            (bbox, _) = features[best_match]
            return bbox
        else:
            print("Could not find the object")
            # HRI.TextToSpeech("I could not find the object you requested. Please make sure it is present.")
            # Maybe start grabbing random objects here to see if it shows up?


    def learn_new_object(self, entity):
        # features = vision.getBoundingBoxesWithFeatures()
        features = np.array([0.34529281, 0.26564698, 0.66764128, 0.0916638, 0.60056184]) # TODO remove
        self.db.insert_feature(entity, features)
        print("New object learnt: ", entity)
        # HRI.TextToSpeech("I have now learned the features of the object you presented to me.")

    def embedding_distance(self, features_1, features_2):
        return np.linalg.norm(features_1 - features_2)

    def is_same_object(self, features_1, features_2, threshold=1):
        return self.embedding_distance(features_1, features_2) < threshold

    def find_best_match(self, index, distances) -> Optional[tuple]:
        if isinstance(index, np.ndarray):
            index = index.tolist()
        if isinstance(distances, np.ndarray):
            distances = distances.tolist()
        if len(index) != len(distances):
            raise ValueError("list_of_tuples should be same length as distances")
        if len(index) == 0 or len(distances) == 0:
            return None
        min_distance = 3
        best_index = None
        for i, distance in enumerate(distances):
            if distance < min_distance:
                min_distance = distance
                best_index = i
        return index[best_index]

    def update_features(self, entity):
        db_features = self.db.select(entity)
        # HRI.TextToSpeech("please place the object you want me to update features for on the table")
        # Wait for affirmation
        # features = vision.getBoundingBoxesWithFeatures()
        features = 0 # TODO remove
        new_features = db_features * 0.9 + features * 0.10 # TODO discuss this
        self.db.update(entity, new_features)

if __name__ == "__main__":
    test = Grounding()
    #test.learnNewObject("black cover")
    test.find_object('red cover')
    #test.update_features("blue cover")