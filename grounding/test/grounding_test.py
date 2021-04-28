from grounding_lib.grounding import Grounding, GroundingReturn, ErrorType
from grounding_lib.spatial import SpatialRelation, StatusEnum
from database_handler.database_handler import DatabaseHandler
import numpy as np
import unittest
from unittest.mock import MagicMock, Mock
from ner_lib.command_builder import CommandBuilder, ObjectEntity, SpatialDescription, SpatialType
from ner_lib.ner import EntityType
from vision_lib.object_info_with_features import ObjectInfoWithFeatures


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
        returned = self.grounding.find_object(object_entity)
        self.assertEqual(returned.error_code, ErrorType.CANT_FIND)

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
        self.db_mock.get_feature = Mock(return_value=None)
        object_name = "albert is cool"
        object_spatial_desc = None
        object_entity = ObjectEntity()
        object_entity.name = object_name
        object_entity.spatial_descriptions = object_spatial_desc
        returned = self.grounding.find_object(object_entity)
        self.assertEqual(returned.error_code, ErrorType.UNKNOWN)  # Should return false if the object is unknown

    def test_spatialPart(self):
        entities = [
            (EntityType.TASK, "pick up"),
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
        self.spatial_mock.locate_specific_object = Mock(return_value=(1, StatusEnum.SUCCESS))
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        returned = self.grounding.find_object(object_entity)

        self.assertTrue(returned.is_success)
        self.assertIsNotNone(returned.object_info)


class LearnObjectIsolatedTest(unittest.TestCase):
    def setUp(self):
        self.skipTest("Not used ATM, needs to be refactored if we include learning objects")
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
        self.skipTest("Feature is not used ATM and needs refactoring if it is implemented")
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
        self.db_mock = Mock()
        self.spatial = SpatialRelation(database_handler=self.db_mock)
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
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_right(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_left(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.OBJECT, "fuse"),
            (EntityType.LOCATION, "left"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[2][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_below(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "below"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[4][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_diagonalRightUp1(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_diagonalRightUp2(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_serial(self):
        entities = [
            (EntityType.TASK, "pick up"),
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
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[0][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_locate_specific_object__static_location__returns_closest_object(self):
        object_entity = ObjectEntity(name="blue cover")
        spatial_description = SpatialDescription(spatial_type=SpatialType.OTHER)
        spatial_description.object_entity.name = "top left corner"
        object_entity.spatial_descriptions.append(spatial_description)
        self.db_mock.get_location_by_name = Mock(return_value=(100, 100, 0))

        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)

        self.assertEqual(expected_index, actual_index)


class SpatialModuleTwoOfEach(unittest.TestCase):
    def setUp(self):
        self.ner_mock = Mock()
        self.db_mock = Mock()
        self.spatial = SpatialRelation(database_handler=self.db_mock)
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
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_right(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_left(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "left"),
            (EntityType.OBJECT, "bottom cover")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[4][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_below(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "below"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[4][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_diagonalRightUp(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "right"),
            (EntityType.OBJECT, "fuse")
        ]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_serial(self):
        entities = [
            (EntityType.TASK, "pick up"),
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
        object_entity = task.objects_to_execute_on[0]
        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)
        self.assertEqual(expected_index, actual_index)

    def test_twoOfReference(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover")]

        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        object_entity = task.objects_to_execute_on[0]
        index, status = self.spatial.locate_specific_object(object_entity, self.objects)

        self.assertEqual(StatusEnum.ERROR_TWO_REF, status)

    def test_locate_specific_object__two_valid_results__returns_list_with_correct_length(self):
        self.skipTest("Contradicts twoOfReference test")
        entities = [
            (EntityType.TASK, "pick up"),
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
        object_entity = task.objects_to_execute_on[0]
        self.assertEqual(2, len(self.spatial.locate_specific_object(object_entity, objects)))

    def test_locate_specific_object__static_location__returns_closest_object(self):
        object_entity = ObjectEntity(name="blue cover")
        spatial_description = SpatialDescription(spatial_type=SpatialType.OTHER)
        spatial_description.object_entity.name = "top left corner"
        object_entity.spatial_descriptions.append(spatial_description)
        self.db_mock.get_location_by_name = Mock(return_value=(100, 100, 0))

        expected_index = self.objects[0][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)

        self.assertEqual(expected_index, actual_index)

    def test_locate_specific_object__static_location_2__returns_closest_object(self):
        object_entity = ObjectEntity(name="blue cover")
        spatial_description = SpatialDescription(spatial_type=SpatialType.OTHER)
        spatial_description.object_entity.name = "top left corner"
        object_entity.spatial_descriptions.append(spatial_description)
        self.db_mock.get_location_by_name = Mock(return_value=(700, 100, 0))

        expected_index = self.objects[1][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)

        self.assertEqual(expected_index, actual_index)

    def test_locate_specific_object__relative_to_object_at_static_location__returns_closest_object(self):
        object_entity = ObjectEntity(name="blue cover")
        spatial_description = SpatialDescription(spatial_type=SpatialType.NEXT_TO)
        spatial_description.object_entity.name = "white cover"
        spatial_description_2 = SpatialDescription(spatial_type=SpatialType.OTHER)
        spatial_description_2.object_entity.name = "top left corner"
        object_entity.spatial_descriptions.append(spatial_description)
        object_entity.spatial_descriptions.append(spatial_description_2)
        self.db_mock.get_location_by_name = Mock(return_value=(100, 100, 0))

        expected_index = self.objects[6][0]
        actual_index, success_enum = self.spatial.locate_specific_object(object_entity, self.objects)

        self.assertEqual(expected_index, actual_index)

    def test_get_location__static_location__returns_static_location(self):
        spatial_description = SpatialDescription(spatial_type=SpatialType.OTHER)
        spatial_description.object_entity.name = "top left corner"
        spatial_descriptions = [spatial_description]
        expected_x, expected_y = 100, 100
        self.db_mock.get_location_by_name = Mock(return_value=(expected_x, expected_y, 0))

        x, y = self.spatial.get_location(spatial_descriptions, self.objects)

        self.assertEqual(expected_x, x)
        self.assertEqual(expected_y, y)

    def test_get_location__object_at_static_location__return_object_center(self):
        objects = [ # copy to make sure setUp changes dont mess up this test
            (0, "blue cover", [100, 400, 100, 300]),
            (1, "blue cover", [700, 1100, 100, 300]),
            (2, "fuse", [100, 400, 800, 1000]),
            (3, "bottom cover", [700, 1100, 800, 1000]),
            (4, "white cover", [100, 400, 1500, 1600]),
            (5, "white cover", [700, 700, 1500, 1600]),
            (6, "blue cover", [0, 100, 1550, 1600])
        ]

        spatial_description = SpatialDescription(spatial_type=SpatialType.NEXT_TO)
        spatial_description.object_entity.name = "white cover"
        spatial_description_2 = SpatialDescription(spatial_type=SpatialType.OTHER)
        spatial_description_2.object_entity.name = "top left corner"
        spatial_descriptions = []
        spatial_descriptions.append(spatial_description)
        spatial_descriptions.append(spatial_description_2)
        expected_bbox = objects[4][2]
        expected_x, expected_y = expected_bbox[1] - (expected_bbox[1] - expected_bbox[0]) / 2, expected_bbox[3] - (expected_bbox[3] - expected_bbox[2]) / 2
        self.db_mock.get_location_by_name = Mock(return_value=(100, 100, 0))

        x, y = self.spatial.get_location(spatial_descriptions, objects)

        self.assertAlmostEqual(expected_x, x)
        self.assertAlmostEqual(expected_y, y)


################################# ISOLATED UNIT TESTS ----- END ##########################################################

################################# INTEGRATION TESTS ----- BEGIN ##########################################################

class FindObjectIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.skipTest("Skip until it is needed")
        self.database_handler = DatabaseHandler("test_grounding.db")
        self.vision_mock = Mock()  # TODO remove this once we have a vision module
        self.spatial_relation_mock = Mock()
        self.grounding = Grounding(vision_controller=self.vision_mock, db=self.database_handler, spatial=self.spatial_relation_mock)
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
        self.skipTest("Skip until it is needed")
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
        self.skipTest("Skip until it is needed")
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
