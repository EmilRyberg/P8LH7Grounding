from ner.command_builder import CommandBuilder


def test_ner(model_path, tag_path):
    cmd_builder = CommandBuilder(model_path, tag_path)
    while True:
        print("Input sentence to test:")
        text = input("> ")
        task = cmd_builder.get_task(text)
        print(task)


if __name__ == "__main__":
    test_ner("ner_1/pytorch_model.bin", "tags.txt")

