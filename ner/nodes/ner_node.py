#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from ner.command_builder import CommandBuilder
from ner.ner import EntityType
from little_helper_interfaces.msg import Task, ObjectEntity, OuterObjectEntity, SpatialDescription
from ner.srv import NER, NERResponse, NERRequest


class DummyNER:
    def get_entities(self, sentence):
        return [
            (EntityType.FIND, "find"),
            (EntityType.COLOUR, "blue"),
            (EntityType.OBJECT, "bottom cover")
        ]


class NERService:
    def __init__(self):
        self.cmd_builder = CommandBuilder("", "", DummyNER())#CommandBuilder("model.bin", "tags.txt")

    def handle_ner_request(self, request: NERRequest):
        result = self.cmd_builder.get_task(request.sentence)
        task = Task(type="take")
        return NERResponse(task=task)

    def ner_server(self):
        rospy.init_node("ner_server")
        s = rospy.Service("ner", NER, self.handle_ner_request)
        rospy.spin()


if __name__ == '__main__':
    service = NERService()
    service.ner_server()

