import unittest
from ner.ner import NER, EntityType
from unittest.mock import MagicMock, Mock
from ner.command_builder import CommandBuilder, PickUpTask, SpatialDescription, ObjectEntity, SpatialType, FindTask


class NERTestCase(unittest.TestCase):
    def setUp(self):
        self.ner = NER("ner_model.bin", "tags.txt")

    def test_get_entities__sentence_with_pick_up_task__returns_all_entities(self):
        sentence = "Please pick up the blue cover next to the black bottom cover"
        entities = self.ner.get_entities(sentence)
        self.assertEqual(len(entities), 6)
        self.assertEqual(entities[0][0], EntityType.TAKE)
        self.assertEqual(entities[1][0], EntityType.COLOUR)
        self.assertEqual(entities[2][0], EntityType.OBJECT)
        self.assertEqual(entities[3][0], EntityType.LOCATION)
        self.assertEqual(entities[4][0], EntityType.COLOUR)
        self.assertEqual(entities[5][0], EntityType.OBJECT)


class CommandBuilderTestCase(unittest.TestCase):
    def setUp(self):
        self.ner_mock = Mock()
        self.cmd_builder = CommandBuilder("", "", self.ner_mock)

    def test_get_task__entities_with_pick_up_task__returns_pick_up_task_and_locations(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        self.assertIsInstance(task, PickUpTask)
        self.assertIsNotNone(task.object_to_pick_up)
        self.assertEqual(task.object_to_pick_up.name, "blue cover")
        self.assertIsNotNone(task.object_to_pick_up.spatial_descriptor)
        self.assertEqual(task.object_to_pick_up.spatial_descriptor.spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.object_to_pick_up.spatial_descriptor.object_entity)
        self.assertEqual(task.object_to_pick_up.spatial_descriptor.object_entity.name, "black bottom cover")
        self.assertIsNotNone(task.object_to_pick_up.spatial_descriptor.object_entity.spatial_descriptor)
        self.assertEqual(task.object_to_pick_up.spatial_descriptor.object_entity.spatial_descriptor.spatial_type, SpatialType.ABOVE)
        self.assertIsNotNone(task.object_to_pick_up.spatial_descriptor.object_entity.spatial_descriptor.object_entity)
        self.assertEqual(task.object_to_pick_up.spatial_descriptor.object_entity.spatial_descriptor.object_entity.name,
                         "bottom cover")

    def test_get_task__entities_with_find_task__returns_find_task_and_locations(self):
        entities = [
            (EntityType.FIND, "find"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "yellow"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        self.assertIsInstance(task, FindTask)
        self.assertIsNotNone(task.object_to_find)
        self.assertEqual(task.object_to_find.name, "blue cover")
        self.assertIsNotNone(task.object_to_find.spatial_descriptor)
        self.assertEqual(task.object_to_find.spatial_descriptor.spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.object_to_find.spatial_descriptor.object_entity)
        self.assertEqual(task.object_to_find.spatial_descriptor.object_entity.name, "yellow bottom cover")
        self.assertIsNone(task.object_to_find.spatial_descriptor.object_entity.spatial_descriptor)


class NERIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.cmd_builder = CommandBuilder("ner_model.bin", "tags.txt")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_task_and_locations(self):
        task = self.cmd_builder.get_task("Please pick up the blue cover that is next to the black bottom cover which is above a bottom cover")
        self.assertIsInstance(task, PickUpTask)
        self.assertIsNotNone(task.object_to_pick_up)
        self.assertEqual(task.object_to_pick_up.name, "blue cover")
        self.assertIsNotNone(task.object_to_pick_up.spatial_descriptor)
        self.assertEqual(task.object_to_pick_up.spatial_descriptor.spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.object_to_pick_up.spatial_descriptor.object_entity)
        self.assertEqual(task.object_to_pick_up.spatial_descriptor.object_entity.name, "black bottom cover")
        self.assertIsNotNone(task.object_to_pick_up.spatial_descriptor.object_entity.spatial_descriptor)
        self.assertEqual(task.object_to_pick_up.spatial_descriptor.object_entity.spatial_descriptor.spatial_type, SpatialType.ABOVE)
        self.assertIsNotNone(task.object_to_pick_up.spatial_descriptor.object_entity.spatial_descriptor.object_entity)
        self.assertEqual(task.object_to_pick_up.spatial_descriptor.object_entity.spatial_descriptor.object_entity.name,
                         "bottom cover")

    def test_get_task__sentence_with_find_task__returns_pick_up_task_and_locations(self):
        task = self.cmd_builder.get_task("Please find the blue cover that is next to the yellow bottom cover")
        self.assertIsInstance(task, FindTask)
        self.assertIsNotNone(task.object_to_find)
        self.assertEqual(task.object_to_find.name, "blue cover")
        self.assertIsNotNone(task.object_to_find.spatial_descriptor)
        self.assertEqual(task.object_to_find.spatial_descriptor.spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.object_to_find.spatial_descriptor.object_entity)
        self.assertEqual(task.object_to_find.spatial_descriptor.object_entity.name, "yellow bottom cover")
        self.assertIsNone(task.object_to_find.spatial_descriptor.object_entity.spatial_descriptor)