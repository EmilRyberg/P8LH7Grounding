import unittest
from ner_lib.ner import NER, EntityType
from unittest.mock import Mock
from ner_lib.command_builder import CommandBuilder, SpatialType, Task, TaskType


class NERTestCase(unittest.TestCase):
    def setUp(self):
        self.ner = NER("ner_model.bin", "../scripts/NER_4/tags.txt")

    def test_get_entities__sentence_with_pick_up_task__returns_all_entities(self):
        sentence = "Please pick up the blue cover next to the black bottom cover"
        entities = self.ner.get_entities(sentence)
        self.assertEqual(6, len(entities))
        self.assertEqual(entities[0][0], EntityType.TASK)
        self.assertEqual(entities[1][0], EntityType.COLOUR)
        self.assertEqual(entities[2][0], EntityType.OBJECT)
        self.assertEqual(entities[3][0], EntityType.LOCATION)
        self.assertEqual(entities[4][0], EntityType.COLOUR)
        self.assertEqual(entities[5][0], EntityType.OBJECT)


class CommandBuilderTestCase(unittest.TestCase):
    def setUp(self):
        self.ner_mock = Mock()
        self.cmd_builder = CommandBuilder(self.ner_mock)

    def test_get_task__entities_with_pick_up_task__returns_pick_up_task_and_locations(self):
        entities = [
            (EntityType.TASK, "pick up"),
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

        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual("blue cover", task.objects_to_execute_on[0].name)
        self.assertEqual(2, len(task.objects_to_execute_on[0].spatial_descriptions))
        self.assertEqual(SpatialType.NEXT_TO, task.objects_to_execute_on[0].spatial_descriptions[0].spatial_type)
        self.assertIsNotNone(task.objects_to_execute_on[0].spatial_descriptions[0].object_entity)
        self.assertEqual("black bottom cover", task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)
        self.assertEqual(SpatialType.ABOVE, task.objects_to_execute_on[0].spatial_descriptions[1].spatial_type)
        self.assertIsNotNone(task.objects_to_execute_on[0].spatial_descriptions[1].object_entity)
        self.assertEqual("bottom cover", task.objects_to_execute_on[0].spatial_descriptions[1].object_entity.name)

    def test_get_task__entities_with_task__returns_task_with_name(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual("pick up", task.name)

    def test_get_task__entities_with_pick_up_task__returns_pick_up_task_with_two_objects(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual("blue cover", task.objects_to_execute_on[0].name)
        self.assertEqual("black bottom cover", task.objects_to_execute_on[1].name)

    def test_get_task__entities_with_pick_up_task__returns_pick_up_task_with_multiple_objects(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.COLOUR, "green"),
            (EntityType.OBJECT, "fuse"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.COLOUR, "red"),
            (EntityType.OBJECT, "fuse")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual("blue cover", task.objects_to_execute_on[0].name)
        self.assertEqual("green fuse", task.objects_to_execute_on[1].name)
        self.assertEqual("black bottom cover", task.objects_to_execute_on[2].name)
        self.assertEqual("red fuse", task.objects_to_execute_on[3].name)

    def test_get_task__entities_with_pick_up_task__returns_pick_up_task_with_multiple_objects_with_spatial_description(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "green"),
            (EntityType.OBJECT, "fuse"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.COLOUR, "red"),
            (EntityType.OBJECT, "fuse")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual("blue cover", task.objects_to_execute_on[0].name)
        self.assertEqual("green fuse", task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)
        self.assertEqual("black bottom cover", task.objects_to_execute_on[1].name)
        self.assertEqual("red fuse", task.objects_to_execute_on[2].name)

    def test_get_task__entities_with_pick_up_task__returns_pick_up_task_with_multiple_objects_with_multiple_spatial_descriptions(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "green"),
            (EntityType.OBJECT, "fuse"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "red"),
            (EntityType.OBJECT, "fuse")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual("blue cover", task.objects_to_execute_on[0].name)
        self.assertEqual("green fuse", task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)
        self.assertEqual("black bottom cover", task.objects_to_execute_on[1].name)
        self.assertEqual("red fuse", task.objects_to_execute_on[1].spatial_descriptions[0].object_entity.name)

    def test_get_task__entities_with_pick_up_task_with_no_spatial_relations__returns_pick_up_task(self):
        entities = [
            (EntityType.TASK, "pick up"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual("blue cover", task.objects_to_execute_on[0].name)
        self.assertEqual(0, len(task.objects_to_execute_on[0].spatial_descriptions))

    def test_get_task__entities_with_find_task__returns_find_task_and_locations(self):
        entities = [
            (EntityType.TASK, "find"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "yellow"),
            (EntityType.OBJECT, "bottom cover")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual("blue cover", task.objects_to_execute_on[0].name)
        self.assertEqual(1, len(task.objects_to_execute_on[0].spatial_descriptions))
        self.assertEqual(SpatialType.NEXT_TO, task.objects_to_execute_on[0].spatial_descriptions[0].spatial_type)
        self.assertIsNotNone(task.objects_to_execute_on[0].spatial_descriptions[0].object_entity)
        self.assertEqual("yellow bottom cover", task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)

    def test_get_task__entites_with_static_location__returns_task_with_correct_spatial_type(self):
        entities = [
            (EntityType.TASK, "place"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "top left corner"),
            (EntityType.OBJECT, "table")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual(SpatialType.OTHER, task.objects_to_execute_on[0].spatial_descriptions[0].spatial_type)

    def test_get_task__entites_with_static_location__returns_task_with_correct_spatial_name(self):
        entities = [
            (EntityType.TASK, "place"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "top left corner"),
            (EntityType.OBJECT, "table")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual("top left corner", task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)

    def test_get_task__entites_with_relative_and_static_location__returns_task_with_correct_spatial_types(self):
        entities = [
            (EntityType.TASK, "place"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "top left corner"),
            (EntityType.OBJECT, "table")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual(SpatialType.NEXT_TO, task.objects_to_execute_on[0].spatial_descriptions[0].spatial_type)
        self.assertEqual(SpatialType.OTHER, task.objects_to_execute_on[0].spatial_descriptions[1].spatial_type)

    def test_get_task__entites_with_relative_and_static_location__returns_task_with_correct_spatial_names(self):
        entities = [
            (EntityType.TASK, "place"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "top left corner"),
            (EntityType.OBJECT, "table")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual("white cover", task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)
        self.assertEqual("top left corner", task.objects_to_execute_on[0].spatial_descriptions[1].object_entity.name)

    def test_get_task_entities__no_main_object__returns_object_with_spatial_descriptions(self):
        # example Place it in the top left corner
        entities = [
            (EntityType.TASK, "place"),
            (EntityType.LOCATION, "top left corner"),
            (EntityType.OBJECT, "table")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = self.cmd_builder.get_task("Dummy sentence")

        self.assertEqual("", task.objects_to_execute_on[0].name)
        self.assertEqual("top left corner", task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)

    def test_add_entities_to_task__objects_in_entity_list__adds_object_to_task(self):
        entities = [
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = Task("pick")
        task.task_type = TaskType.PICK
        self.cmd_builder.add_entities_to_task(task, "Dummy sentence")

        self.assertEqual("blue cover", task.objects_to_execute_on[0].name)

    def test_add_entities_to_task__sub_task_in_entity_list__adds_subtask_to_task(self):
        entities = [
            (EntityType.TASK, "place"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "white"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "top left corner"),
            (EntityType.OBJECT, "table")
        ]
        self.ner_mock.get_entities = Mock(return_value=entities)
        task = Task("pick")
        task.task_type = TaskType.PICK
        self.cmd_builder.add_entities_to_task(task, "Dummy sentence")

        self.assertEqual("place", task.child_tasks[0].name)
        self.assertEqual("white cover", task.child_tasks[0].objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)
        self.assertEqual("top left corner", task.child_tasks[0].objects_to_execute_on[0].spatial_descriptions[1].object_entity.name)


class NERIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.ner = NER("ner_model.bin", "../scripts/NER_4/tags.txt")
        self.cmd_builder = CommandBuilder(self.ner)

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_task_and_locations(self):
        task = self.cmd_builder.get_task("Please pick up the blue cover that is next to the black bottom cover which is above a bottom cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "blue cover")
        self.assertEqual(len(task.objects_to_execute_on[0].spatial_descriptions), 2)
        self.assertEqual(task.objects_to_execute_on[0].spatial_descriptions[0].spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.objects_to_execute_on[0].spatial_descriptions[0].object_entity)
        self.assertEqual("black bottom cover", task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)
        self.assertEqual(task.objects_to_execute_on[0].spatial_descriptions[1].spatial_type, SpatialType.ABOVE)
        self.assertIsNotNone(task.objects_to_execute_on[0].spatial_descriptions[1].object_entity)
        self.assertEqual("bottom cover", task.objects_to_execute_on[0].spatial_descriptions[1].object_entity.name)

    def test_get_task__sentence_with_find_task__returns_pick_up_task_and_locations(self):
        task = self.cmd_builder.get_task("Please find the blue cover that is next to the yellow bottom cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "blue cover")
        self.assertEqual(len(task.objects_to_execute_on[0].spatial_descriptions), 1)
        self.assertEqual(task.objects_to_execute_on[0].spatial_descriptions[0].spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.objects_to_execute_on[0].spatial_descriptions[0].object_entity)
        self.assertEqual(task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name, "yellow bottom cover")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_task_and_locations(self):
        task = self.cmd_builder.get_task(
            "Please pick up the blue cover that is next to the black bottom cover which is above a bottom cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "blue cover")
        self.assertEqual(len(task.objects_to_execute_on[0].spatial_descriptions), 2)
        self.assertEqual(task.objects_to_execute_on[0].spatial_descriptions[0].spatial_type, SpatialType.NEXT_TO)
        self.assertIsNotNone(task.objects_to_execute_on[0].spatial_descriptions[0].object_entity)
        self.assertEqual("black bottom cover",
                         task.objects_to_execute_on[0].spatial_descriptions[0].object_entity.name)
        self.assertEqual(task.objects_to_execute_on[0].spatial_descriptions[1].spatial_type, SpatialType.ABOVE)
        self.assertIsNotNone(task.objects_to_execute_on[0].spatial_descriptions[1].object_entity)
        self.assertEqual("bottom cover", task.objects_to_execute_on[0].spatial_descriptions[1].object_entity.name)

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test1(self):
        task = self.cmd_builder.get_task(
            "Please pick up the black bottom cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "black bottom cover")
        self.assertTrue(task.name == "pick up")


    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test2(self):
        task = self.cmd_builder.get_task(
            "Please pick up the screwdriver")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "screwdriver")
        self.assertEqual(task.name,"get")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test3(self):
        task = self.cmd_builder.get_task(
            "Please move the hammer")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "hammer")
        self.assertEqual(task.name,"move")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test4(self):
        task = self.cmd_builder.get_task(
            "Please pick up the fuse next to the book")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "fuse")
        self.assertEqual(task.name, "pick up")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test5(self):
        task = self.cmd_builder.get_task(
            "Please pick up the green cover on top of the table")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "green cover")
        self.assertEqual(task.name, "pick up")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test6(self):
        task = self.cmd_builder.get_task(
            "Please place the blue cover on top of the black bottom cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "blue cover")
        self.assertEqual(task.name, "place")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test7(self):
        task = self.cmd_builder.get_task(
            "Please find the white bottom cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "white bottom cover")
        self.assertEqual(task.name, "find")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test8(self):
        task = self.cmd_builder.get_task(
            "Please find the fuse")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "fuse")
        self.assertEqual(task.name, "find")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test9(self):
        task = self.cmd_builder.get_task(
            "Please weld the cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "cover")
        self.assertEqual(task.name, "weld")


    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test10(self):
        task = self.cmd_builder.get_task(
            "Please put the fuse on top of the white cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "fuse")
        self.assertEqual(task.name, "put")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test11(self):
        task = self.cmd_builder.get_task(
            "Please place the black bottom cover on top of the white cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "black bottom cover")
        self.assertEqual(task.name, "place")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test12(self):
        task = self.cmd_builder.get_task(
            "Please place the white cover on top of the black bottom cover")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "white cover")
        self.assertEqual(task.name, "place")


    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test13(self):
        task = self.cmd_builder.get_task(
            "Hey robot, could you destroy the car for me?")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "car")
        self.assertEqual(task.name, "destroy")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test14(self):
        task = self.cmd_builder.get_task(
            "Hey robot, could you fly to the moon?")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "moon")
        self.assertEqual(task.name, "fly")

    def test_get_task__sentence_with_pick_up_task__returns_pick_up_test15(self):
        task = self.cmd_builder.get_task(
            "Hey robot, could you turn the tool?")
        self.assertIsNotNone(task.objects_to_execute_on[0])
        self.assertEqual(task.objects_to_execute_on[0].name, "tool")
        self.assertEqual(task.name, "turn")
