from ner_lib.ner import NER, EntityType


def test_ner(model_path, tag_path):
    ner = NER(model_path, tag_path)
    while True:
        print("Input sentence to test:")
        text = input("> ")
        entities = ner.get_entities(text)
        print("Entities", entities)


if __name__ == "__main__":
    test_ner("NER_5/pytorch_model.bin", "NER_5/tags.txt")

