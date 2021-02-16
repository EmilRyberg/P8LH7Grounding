import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import numpy as np

# unique tags {'O', 'I-object', 'B-color', 'B-object'}

id2tag = [
    "O",
    "B-object",
    "I-object",
    "B-color",
    "B-grasp",
    "I-grasp",
    "B-find",
    "I-find",
    "B-location",
    "I-location"
]


def test_model(model_path):
    model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    run = True
    while run:
        print("Input sentence to test:")
        text = input("> ")
        encoded = tokenizer(text, is_split_into_words=False, return_offsets_mapping=True, padding=True,
                                    truncation=True, return_tensors="pt")
        output = model(encoded.input_ids, encoded.attention_mask)
        logits = output.logits
        logits_softmax = torch.nn.Softmax(dim=2)(logits).detach().cpu()
        entities = []
        for token_index in range(logits_softmax.shape[1]):
            max_id = torch.argmax(logits_softmax[0, token_index, :]).numpy()
            max_id_value = logits_softmax[0, token_index, max_id].numpy()
            current_offsets = encoded.offset_mapping[0, token_index, :]
            if current_offsets[0] == 0 and current_offsets[1] == 0:
                continue
            word = text[current_offsets[0]:current_offsets[1]]
            if id2tag[max_id] == "O":
                continue
            entities.append((word, id2tag[max_id], max_id_value))
        print("Found entities:")
        for (word, tag, conf) in entities:
            print(f"'{word}': {tag} (conf: {conf*100.0:.4f}%)")


if __name__ == "__main__":
    test_model("test/pytorch_model.bin")