from grounding.grounding import Grounding, GroundingReturn, ErrorType
from grounding.spatial import SpatialRelation
import numpy as np
import unittest
from unittest.mock import MagicMock, Mock
from ner_lib.command_builder import CommandBuilder, ObjectEntity
from ner_lib.ner import EntityType
from vision_lib.vision_controller import VisionController, ObjectInfoWithFeatures


################################# ISOLATED UNIT TESTS ----- BEGIN ##########################################################

class FindObjectIsolatedTest(unittest.TestCase):
    def setUp(self):
        self.ner_mock = Mock()
        self.db_mock = Mock()
        self.spatial_mock = Mock()
        self.vision_mock = Mock()
        self.returned = GroundingReturn()
        self.cmd_builder = CommandBuilder("", "", self.ner_mock)
        self.grounding = Grounding(db=self.db_mock, vision_controller=self.vision_mock, spatial=self.spatial_mock)
        self.objects = [  # bbox = [x1, x2, y1, y2] and images spans from 0,0 to 1500,2000
            (0, "black cover", [100, 400, 100, 300]),
            (1, "blue cover", [700, 1100, 100, 300]),
            (2, "fuse", [100, 400, 800, 1000]),
            (3, "bottom cover", [700, 1100, 800, 1000]),
            (4, "white cover", [100, 400, 1500, 1600]),
            (5, "blue cover", [700, 700, 1500, 1600])
        ]

    def test_failFindObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        self.db_mock.get_feature = Mock(return_value=np.array([5, 5, 5, 5, 5]))
        object_name = "black cover"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.find_object(object_entity)
        self.assertEqual(self.returned.error_code, ErrorType.CANT_FIND)

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
        object_info = ObjectInfoWithFeaturesWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        self.db_mock.get_feature = Mock(return_value=None)
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.find_object(object_entity)
        self.assertEqual(self.returned.error_code, ErrorType.UNKNOWN)  # Should return false if the object is unknown

    def test_spatialPart(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        features = []
        feature = [np.array([1, 1, 1, 1, 1]),
                   np.array([1, 1, 1, 1, 1]),
                   np.array([1, 1, 1, 1, 1])
                   ]
        bbox = [np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4]),
                np.array([1, 2, 3, 4])
                ]
        mask = [np.array([4, 3, 2 ,1]),
                np.array([4, 3, 2, 1]),
                np.array([4, 3, 2, 1])
                ]
        cropped_rbg = [np.array([5, 5, 5, 5]),
                       np.array([5, 5, 5, 5]),
                       np.array([5, 5, 5, 5])
                       ]
        for i in range(3):
            object_info = ObjectInfoWithFeatures()
            object_info.features = feature[i]
            object_info.bbox_xxyy = bbox[i]
            object_info.mask_full = mask[i]
            object_info.mask_cropped = cropped_rbg[i]
            features.append(object_info)

        db_features = [
            ("dummy name", np.array([1, 1, 1, 1, 1])),
            ("dummy name", np.array([1, 1, 1, 1, 1])),
            ("dummy name", np.array([1, 1, 1, 1, 1]))
        ]
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        self.db_mock.get_feature = Mock(return_value=np.array([1, 1, 1, 1, 1]))
        self.db_mock.get_all_features = Mock(return_value=db_features)
        self.ner_mock.get_entities = Mock(return_value=entities)
        self.spatial_mock.locate_specific_object = Mock(return_value=1)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.object_to_execute_on
        self.returned = self.grounding.find_object(object_entity)
        self.assertTrue(self.returned.is_success)
        self.assertIsNotNone(self.returned.object_info)


class LearnObjectIsolatedTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.vision_mock = Mock()
        self.grounding = Grounding(db=self.db_mock, vision_controller=self.vision_mock)
        self.returned = GroundingReturn()

    def test_newObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        self.db_mock.get_feature = Mock(return_value=None)
        entity_name = "green cover"  # Make sure this is a new object.
        spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = entity_name
        object_entity.spatial_descriptions = spatial_desc
        self.returned = self.grounding.learn_new_object(object_entity)
        self.assertTrue(self.returned.is_success)

    def test_learnKnownObject(self):
        self.db_mock.get_feature = Mock(return_value=123)
        object_name = "green cover"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.learn_new_object(object_entity)
        self.assertEqual(self.returned.error_code, ErrorType.ALREADY_KNOWN)


class UpdateFeaturesIsolatedTest(unittest.TestCase):
    def setUp(self):
        self.db_mock = Mock()
        self.vision_mock = Mock()
        self.grounding = Grounding(db=self.db_mock, vision_controller=self.vision_mock)
        self.returned = GroundingReturn()

    def test_updateCorrectObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.db_mock.get_feature = Mock(return_value=1)
        self.db_mock.update = Mock(return_value=None)

        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        object_name = "green cover"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.update_features(object_entity)
        self.assertTrue(self.returned.is_success)

    def test_updateUnknownObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.db_mock.get_feature = Mock(return_value=None)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.update_features(object_entity)
        self.assertEqual(self.returned.error_code, ErrorType.UNKNOWN)


class SpatialModuleOneOfEach(unittest.TestCase):
    def setUp(self):
        self.ner_mock = Mock()
        self.spatial = SpatialRelation()
        self.cmd_builder = CommandBuilder("", "", self.ner_mock)

        self.objects = [  # bbox = [x1, x2, y1, y2] and images spans from 0,0 to 1500,2000
            (0, "black cover", [100, 400, 100, 300]),
            (1, "blue cover", [700, 1100, 100, 300]),
            (2, "fuse", [100, 400, 800, 1000]),
            (3, "bottom cover", [700, 1100, 800, 1000]),
            (4, "white cover", [100, 400, 1500, 1600]),
            (5, "green cover", [700, 700, 1500, 1600])
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
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[1][0], self.spatial.locate_specific_object(object_entity, self.objects))

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
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[1][0], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_left(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.OBJECT, "fuse"),
            (EntityType.LOCATION, "left"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[2][0], self.spatial.locate_specific_object(object_entity, self.objects))

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
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[4][0], self.spatial.locate_specific_object(object_entity, self.objects))

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
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[1][0], self.spatial.locate_specific_object(object_entity, self.objects))

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
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[1][0], self.spatial.locate_specific_object(object_entity, self.objects))

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
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[0][0], self.spatial.locate_specific_object(object_entity, self.objects))


class SpatialModuleTwoOfEach(unittest.TestCase):
    def setUp(self):
        self.ner_mock = Mock()
        self.spatial = SpatialRelation()
        self.cmd_builder = CommandBuilder("", "", self.ner_mock)

        self.objects = [  # bbox = [x1, x2, y1, y2] and images spans from 0,0 to 1500,2000
            (0, "blue cover", [100, 400, 100, 300]),
            (1, "blue cover", [700, 1100, 100, 300]),
            (2, "fuse", [100, 400, 800, 1000]),
            (3, "bottom cover", [700, 1100, 800, 1000]),
            (4, "white cover", [100, 400, 1500, 1600]),
            (5, "white cover", [700, 700, 1500, 1600]),
            (6, "blue cover", [0, 100, 1550, 1600])
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
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[1][0], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_right(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[1][0], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_left(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "left"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[4][0], self.spatial.locate_specific_object(object_entity, self.objects))

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
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[4][0], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_diagonalRightUp(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[1][0], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_serial(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "fuse"),
            (EntityType.LOCATION, "left"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.object_to_execute_on
        self.assertEqual(self.objects[1][0], self.spatial.locate_specific_object(object_entity, self.objects))

    def test_twoOfReference(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover")]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.object_to_execute_on
        self.assertEqual(-1, self.spatial.locate_specific_object(object_entity, self.objects))

    def test_locate_specific_object__two_valid_results__returns_list_with_correct_length(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover")]

        objects = [  # bbox = [x1, x2, y1, y2] and images spans from 0,0 to 1500,2000
            (0, "white cover", [600, 800, 600, 700]),
            (1, "blue cover", [400, 600, 500, 550]),
            (2, "blue cover", [600, 800, 500, 550]),
            (3, "bottom cover", [700, 1100, 800, 1000]),
            (4, "blue cover", [0, 100, 1550, 1600])
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.object_to_execute_on
        self.assertEqual(2, len(self.spatial.locate_specific_object(object_entity, objects)))


################################# ISOLATED UNIT TESTS ----- END ##########################################################

################################# INTEGRATION TESTS ----- BEGIN ##########################################################

class FindObjectIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.grounding = Grounding()
        self.vision_mock = Mock()  # TODO remove this once we have a vision module
        self.grounding = Grounding(vision_controller=self.vision_mock)
        self.returned = GroundingReturn()

    def test_failFindObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        object_name = "blue cover"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.find_object(object_entity)
        self.assertEqual(self.returned.error_code, ErrorType.CANT_FIND)

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
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        self.returned = self.grounding.find_object(object_entity)
        self.assertEqual(self.returned.error_code, ErrorType.UNKNOWN)


class LearnObjectIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.vision_mock = Mock()
        self.grounding = Grounding(vision_controller=self.vision_mock)
        self.returned = GroundingReturn()

    def test_learnKnownObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        object_name = "black cover"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.learn_new_object(object_entity)
        self.assertEqual(self.returned.error_code, ErrorType.ALREADY_KNOWN)

    def test_newObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        entity_name = "green cover"  # Make sure this is a new object.
        spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = entity_name
        object_entity.spatial_descriptions = spatial_desc
        self.returned = self.grounding.learn_new_object(object_entity)
        self.assertTrue(self.returned.is_success)


class UpdateFeaturesIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.vision_mock = Mock()
        self.grounding = Grounding(vision_controller=self.vision_mock)
        self.returned = GroundingReturn()

    def test_updateCorrectObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        object_name = "black cover"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.update_features(object_entity)
        self.assertTrue(self.returned.is_success)

    def test_updateUnknownObject(self):
        object_info = ObjectInfoWithFeatures()
        features = []
        object_info.features = np.array([1, 1, 1, 1, 1])
        object_info.bbox_xxyy = np.array([1, 2, 3, 4])
        object_info.mask_full = np.array([4, 3, 2 ,1])
        object_info.mask_cropped = np.array([5, 5, 5, 5])
        features.append(object_info)
        self.vision_mock.get_masks_with_features = Mock(return_value=features)
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        self.returned = self.grounding.update_features(object_entity)
        self.assertEqual(self.returned.error_code, ErrorType.UNKNOWN)

################################# INTEGRATION TESTS ----- END ##########################################################
