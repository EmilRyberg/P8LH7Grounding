#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from ner.command_builder import CommandBuilder, PickUpTask, FindTask
from ner.ner import EntityType
from little_helper_interfaces.msg import Task, ObjectEntity, OuterObjectEntity, SpatialDescription, OuterTask
from ner.srv import NER, NERResponse, NERRequest


class DummyNER:
    def get_entities(self, sentence):
        return [
            (EntityType.FIND, "find"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "black"),
            (EntityType.OBJECT, "cover"),
            (EntityType.LOCATION, "above"),
            (EntityType.COLOUR, "yellow"),
            (EntityType.OBJECT, "bottom cover"),
            (EntityType.TAKE, "pick up"),
            (EntityType.COLOUR, "red"),
            (EntityType.OBJECT, "fuse"),
            (EntityType.LOCATION, "next"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "cover")
        ]


class NERService:
    def __init__(self):
        self.cmd_builder = CommandBuilder("", "", DummyNER())#CommandBuilder("model.bin", "tags.txt")

    def create_ros_task(self, task):
        ros_task = Task()
        object1_entity = None
        if isinstance(task, PickUpTask):
            ros_task.type = "pick_up"
            object1_entity = task.object_to_pick_up
        elif isinstance(task, FindTask):
            ros_task.type = "find"
            object1_entity = task.object_to_find
        ros_task.object1 = OuterObjectEntity(name=object1_entity.name, spatial_descriptions=[])
        for spatial_description in object1_entity.spatial_descriptions:
            ros_task.object1.spatial_descriptions.append(SpatialDescription(spatial_type=spatial_description.spatial_type.value,
                                                                        object_entity=ObjectEntity(
                                                                            name=spatial_description.object_entity.name)))
        return ros_task

    def handle_ner_request(self, request: NERRequest):
        task = self.cmd_builder.get_task(request.sentence)
        ros_task = self.create_ros_task(task)
        ros_outer_task = OuterTask(type=ros_task.type, object1=ros_task.object1, object2=ros_task.object2, child_tasks=[])

        if len(task.child_tasks) > 0:
            for child_task in task.child_tasks:
                ros_task = self.create_ros_task(child_task)
                ros_outer_task.child_tasks.append(ros_task)
        return NERResponse(task=ros_outer_task)

    def ner_server(self):
        rospy.init_node("ner_server")
        s = rospy.Service("ner", NER, self.handle_ner_request)
        rospy.spin()


if __name__ == '__main__':
    service = NERService()
    service.ner_server()

