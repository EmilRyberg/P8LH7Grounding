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
        self.assertIsNotNone(task.object_to_execute_on)
        self.assertEqual("blue cover", task.object_to_execute_on.name)
        self.assertEqual(2, len(task.object_to_execute_on.spatial_descriptions))
        self.assertEqual(SpatialType.NEXT_TO, task.object_to_execute_on.spatial_descriptions[0].spatial_type)
        self.assertIsNotNone(task.object_to_execute_on.spatial_descriptions[0].object_entity)
        self.assertEqual("black bottom cover", task.object_to_execute_on.spatial_descriptions[0].object_entity.name)
        self.assertEqual(SpatialType.ABOVE, task.object_to_execute_on.spatial_descriptions[1].spatial_type)
        self.assertIsNotNone(task.object_to_execute_on.spatial_descriptions[1].object_entity)
        self.assertEqual("bottom cover", task.object_to_execute_on.spatial_descriptions[1].object_entity.name)

    def test_get_task__entities_with_pick_up_task_with_no_spatial_relations__returns_pick_up_task(self):
        entities = [
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        self.assertIsInstance(task, PickUpTask)
        self.assertIsNotNone(task.object_to_execute_on)
        self.assertEqual("blue cover", task.object_to_execute_on.name)
        self.assertEqual(0, len(task.object_to_execute_on.spatial_descriptions))

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
        self.assertIsNotNone(task.object_to_execute_on)
        self.assertEqual("blue cover", task.object_to_execute_on.name)
        self.assertEqual(1, len(task.object_to_execute_on.spatial_descriptions))
        self.assertEqual(SpatialType.NEXT_TO, task.object_to_execute_on.spatial_descriptions[0].spatial_type)
        self.assertIsNotNone(task.object_to_execute_on.spatial_descriptions[0].object_entity)
        self.assertEqual("yellow bottom cover", task.object_to_execute_on.spatial_descriptions[0].object_entity.name)


class NERIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.cmd_builder = CommandBuilder("ner_model.bin", "tags.txt")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_task_and_locations(self):
        task = self.cmd_builder.get_task("Please pick up the blue cover that is next to the black bottom cover which is above a bottom cover")
        self.assertIsInstance(task, PickUpTask)
        self.assertIsNotNone(task.object_to_execute_on)
        self.assertEqual(task.object_to_execute_on.name, "blue cover")
        self.assertEqual(len(task.object_to_execute_on.spatial_descriptions), 2)
        self.assertEqual(task.object_to_execute_on.spatial_descriptions[0].spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.object_to_execute_on.spatial_descriptions[0].object_entity)
        self.assertEqual( "black bottom cover", task.object_to_execute_on.spatial_descriptions[0].object_entity.name)
        self.assertEqual(task.object_to_execute_on.spatial_descriptions[1].spatial_type, SpatialType.ABOVE)
        self.assertIsNotNone(task.object_to_execute_on.spatial_descriptions[1].object_entity)
        self.assertEqual("bottom cover", task.object_to_execute_on.spatial_descriptions[1].object_entity.name)

    def test_get_task__sentence_with_find_task__returns_pick_up_task_and_locations(self):
        task = self.cmd_builder.get_task("Please find the blue cover that is next to the yellow bottom cover")
        self.assertIsInstance(task, FindTask)
        self.assertIsNotNone(task.object_to_execute_on)
        self.assertEqual(task.object_to_execute_on.name, "blue cover")
        self.assertEqual(len(task.object_to_execute_on.spatial_descriptions), 1)
        self.assertEqual(task.object_to_execute_on.spatial_descriptions[0].spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.object_to_execute_on.spatial_descriptions[0].object_entity)
        self.assertEqual(task.object_to_execute_on.spatial_descriptions[0].object_entity.name, "yellow bottom cover")
