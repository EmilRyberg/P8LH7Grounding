import database_handler
import numpy as np
from typing import Optional

class Grounding:
    def __init__(self):
        self.db=database_handler.DatabaseHandler()

    def find_object(self, object_entity):
        (id, name, spatial_desc) = object_entity
        found_object = False
        features_below_threshold = []
        distances = []
        db_features = self.db.get_feature(name)
        if db_features is None:
            affirmation = False
            # HRI.TextToSpeech("I dont know the object you asked about, do you want me to learn it?")
            # affirmation = NLP.AffirmationCheck
            if affirmation:
                self.learnNewObject(name)
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

        if not found_object:
            print("Could not find the object")
            # HRI.TextToSpeech("I could not find the object you requested. Please make sure it is present.")
            # Maybe start grabbing random objects here to see if it shows up?
            return

        if len(features_below_threshold)>1:
            if spatial_desc is None:
                # HRI.TextToSpeech("I have found more than one of the requested objects. Which one do you want me to pick up?")
                new_spatial_desc = None # TODO add NLP link to get a new spatial descriptor.. 0 = don't care
            if spatial_desc is not None:
                new_object_entity = (id, name, new_spatial_desc)
                object_info = self.find_object_with_spatial_desc(new_object_entity)
                return object_info

        # This part of the code will be executed if there is only 1 of the requested objects in the scene or
        # if the user does not care about what part is picked up.
        best_match = self.find_best_match(features_below_threshold, distances)
        (bbox, _) = features[best_match]
        object_info = (id, bbox, name)
        return object_info

    def find_object_with_spatial_desc(self, object_entity):
        objects = []
        db_objects = self.db.get_all_features()
        # features = vision.getBoundingBoxesWithFeatures()
        features = []
        features_below_threshold = []
        distances = []

        for i, (name, db_features) in enumerate(db_objects):
            for id, (bbox, feature) in enumerate(features):
                distance = self.embedding_distance(db_features, feature)
                is_below_threshold = self.is_same_object(db_features, feature, threshold=0.8) # TODO update threshold
                if is_below_threshold:
                    distances.append(distance)
                    features_below_threshold.append(id)
                    objects.append((id, bbox, name))

        #object_info = spatial_relations.locateSpecificObject(object_entity, objects)
        object_info = (0, 0, 0)
        return object_info

    def learn_new_object(self, entity):
        # features = vision.getBoundingBoxesWithFeatures()
        features = np.array([0.34529281, 0.26564698, 0.66764128, 0.0916638, 0.60056184]) # TODO remove
        self.db.insert_feature(entity, features)
        print("New object learnt: ", entity)
        # HRI.TextToSpeech("I have now learned the features of the object you presented to me.")

    def update_features(self, entity):
        db_features = self.db.get_feature(entity)
        # HRI.TextToSpeech("please place the object you want me to update features for on the table")
        # Wait for affirmation
        # features = vision.getBoundingBoxesWithFeatures()
        features = 0  # TODO remove
        new_features = db_features * 0.9 + features * 0.10  # TODO discuss this
        self.db.update(entity, new_features)

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


if __name__ == "__main__":
    test = Grounding()
    #test.learnNewObject("black cover")
    test.find_object('red cover')
    #test.update_features("blue cover")