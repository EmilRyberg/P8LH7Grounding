from ner_lib.command_builder import CommandBuilder
from ner_lib.ner import NER


def test_ner(model_path, tag_path):
    ner = NER(model_path, tag_path)
    cmd_builder = CommandBuilder(ner)
    while True:
        print("Input sentence to test:")
        text = input("> ")
        task = cmd_builder.get_task(text)
        print(task)


if __name__ == "__main__":
    test_ner("NER_5/pytorch_model.bin", "NER_5/tags.txt")

