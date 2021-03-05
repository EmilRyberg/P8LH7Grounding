from grounding.grounding import Grounding
import numpy as np
import unittest
from unittest.mock import MagicMock, Mock

################################# ISOLATED UNIT TESTS ----- BEGIN ##########################################################

class FindObjectIsolatedTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.grounding = Grounding(db = self.db_mock)

    def test_failFindObject(self):
        self.db_mock.get_feature = Mock(return_value=np.array([5, 5, 5, 5, 5]))
        object_id = 3
        object_name = "black cover"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        found = self.grounding.find_object(object_entity)
        self.assertFalse(found)

    def test_bestMatch(self):
        indexes = [1, 2, 3, 4, 5]
        distances = [0.78, 2, 1.5, 0.23, 0.1]
        best_index = self.grounding.find_best_match(indexes, distances)
        self.assertEqual(5, best_index)

    def test_sameObject(self):
        features_1 = np.array([0.35, 0.88, 0.20, 0.30])
        features_2 = np.array([0.30, 0.85, 0.17, 0.32])
        self.assertTrue(self.grounding.is_same_object(features_1, features_2))

    def test_unknownObject(self):
        self.db_mock.get_feature = Mock(return_value=None)
        object_id = 3
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        self.assertFalse(self.grounding.find_object(object_entity)) #Should return false if the object is unknown

class LearnObjectIsolatedTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.grounding = Grounding(db = self.db_mock)

    def test_newObject(self):
        self.db_mock.get_feature = Mock(return_value=None)
        id = 1
        entity_name = "green cover"  # Make sure this is a new object.
        spatial_desc = None
        object_entity = (id, entity_name, spatial_desc)
        self.grounding.learn_new_object(object_entity)
        self.db_mock.get_feature = Mock(
            return_value=np.array([1, 1, 1, 1, 1]))
        (id, bbox, name) = self.grounding.find_object(object_entity)
        self.assertEqual(entity_name, name)

    def test_learnKnownObject(self):
        self.db_mock.get_feature = Mock(return_value=123)
        object_id = 3
        object_name = "green cover"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        self.assertEqual(self.grounding.learn_new_object(object_entity), "known")

class UpdateFeaturesIsolatedTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.grounding = Grounding(db = self.db_mock)

    def test_updateCorrectObject(self):
        self.db_mock.get_feature = Mock(return_value=1)
        self.db_mock.update = Mock(return_value=None)
        object_id = 3
        object_name = "green cover"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        updated_features = self.grounding.update_features(object_entity)
        self.assertIsNotNone(updated_features)
        self.assertNotEqual("unknown", updated_features)

    def test_updateUnknownObject(self):
        self.db_mock.get_feature = Mock(return_value=None)
        object_id = 3
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        self.assertEqual(self.grounding.update_features(object_entity), "unknown")

################################# ISOLATED UNIT TESTS ----- END ##########################################################

################################# INTEGRATION UNIT TESTS ----- BEGIN ##########################################################

class FindObjectIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.grounding = Grounding()

    def test_failFindObject(self):
        object_id = 3
        object_name = "blue cover"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        found = self.grounding.find_object(object_entity)
        self.assertFalse(found)

    def test_bestMatch(self):
        indexes = [1, 2, 3, 4, 5]
        distances = [0.78, 2, 1.5, 0.23, 0.1]
        best_index = self.grounding.find_best_match(indexes, distances)
        self.assertEqual(5, best_index)

    def test_sameObject(self):
        features_1 = np.array([0.35, 0.88, 0.20, 0.30])
        features_2 = np.array([0.30, 0.85, 0.17, 0.32])
        self.assertTrue(self.grounding.is_same_object(features_1, features_2))

    def test_unknownObject(self):
        object_id = 3
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        self.assertFalse(self.grounding.find_object(object_entity))  # Should return false if the object is unknown


class LearnObjectIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.grounding = Grounding()

    def test_learnKnownObject(self):
        object_id = 3
        object_name = "black cover"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        learn_new_object_return = self.grounding.learn_new_object(object_entity)
        self.assertEqual(learn_new_object_return, "known")

    def test_newObject(self):
        id = 1
        entity_name = "green cover"  # Make sure this is a new object.
        spatial_desc = None
        object_entity = (id, entity_name, spatial_desc)
        self.grounding.learn_new_object(object_entity)
        (id, bbox, name) = self.grounding.find_object(object_entity)
        self.assertEqual(entity_name, name)


class UpdateFeaturesIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.grounding = Grounding()

    def test_updateCorrectObject(self):
        object_id = 3
        object_name = "green cover"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        updated_features = self.grounding.update_features(object_entity)
        self.assertIsNotNone(updated_features)
        self.assertNotEqual("unknown", updated_features)

    def test_updateUnknownObject(self):
        object_id = 3
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = (object_id, object_name, object_spatial_desc)
        self.assertEqual(self.grounding.update_features(object_entity), "unknown")

################################# INTEGRATION UNIT TESTS ----- END ##########################################################