from grounding.grounding import Grounding
from scripts.spatial import Spatial_Relations
import numpy as np
import unittest
from unittest.mock import MagicMock, Mock
from ner.ner.command_builder import CommandBuilder, PickUpTask, SpatialDescription, ObjectEntity, SpatialType, FindTask
from ner.ner.ner import NER, EntityType


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

class SpatialModuleIsolatedTest(unittest.TestCase):
    def setUp(self):
        self.ner_mock = Mock()
        self.spatial = Spatial_Relations()
        self.cmd_builder = CommandBuilder("", "", self.ner_mock)

        self.objects = [  # bbox = [x1, x2, y1, y2] and images spans from 0,0 to 1500,2000
            ("black cover", [100, 400, 100, 300]),
            ("blue cover", [700, 1100, 100, 300]),
            ("fuse", [100, 400, 800, 1000]),
            ("bottom cover", [700, 1100, 800, 1000]),
            ("white cover", [100, 400, 1500, 1600]),
            ("green cover", [700, 700, 1500, 1600])
        ]

    def test_above(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = (1, task.object_to_pick_up.name, task.object_to_pick_up.spatial_descriptions)
        self.assertEqual(self.objects[1], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_right(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = (1, task.object_to_pick_up.name, task.object_to_pick_up.spatial_descriptions)
        self.assertEqual(self.objects[1], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_left(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.OBJECT, "fuse"),
            (EntityType.LOCATION, "left"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = (1, task.object_to_pick_up.name, task.object_to_pick_up.spatial_descriptions)
        self.assertEqual(self.objects[2], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_below(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "below"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = (1, task.object_to_pick_up.name, task.object_to_pick_up.spatial_descriptions)
        self.assertEqual(self.objects[4], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_diagonalRightUp1(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = (1, task.object_to_pick_up.name, task.object_to_pick_up.spatial_descriptions)
        self.assertEqual(self.objects[1], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_diagonalRightUp2(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = (1, task.object_to_pick_up.name, task.object_to_pick_up.spatial_descriptions)
        self.assertEqual(self.objects[1], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_serial(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "left"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = (1, task.object_to_pick_up.name, task.object_to_pick_up.spatial_descriptions)
        self.assertEqual(self.objects[0], self.spatial.locate_specific_object(object_entity, self.objects))

################################# ISOLATED UNIT TESTS ----- END ##########################################################

################################# INTEGRATION TESTS ----- BEGIN ##########################################################

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

################################# INTEGRATION TESTS ----- END ##########################################################