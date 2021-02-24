import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import numpy as np
from enum import Enum


class EntityType(Enum):
    COLOUR = "colour"
    OBJECT = "object"
    LOCATION = "location"
    FIND = "find"
    TAKE = "take"


class NER:
    def __init__(self, model_path, tag_path):
        with open(tag_path, "r") as tag_file:
            file_content = tag_file.read().strip()
            self.id_to_tag = file_content.splitlines()
        self.model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(self.id_to_tag))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    def get_entities(self, sentence):
        encoded = self.tokenizer(sentence, is_split_into_words=False, return_offsets_mapping=True, padding=True,
                                    truncation=True, return_tensors="pt")
        output = self.model(encoded.input_ids, encoded.attention_mask)
        logits_softmax = torch.nn.Softmax(dim=2)(output.logits).detach().cpu()
        entities = []
        current_entity_word = ""
        current_entity = ""
        for token_index in range(logits_softmax.shape[1]):
            max_id = torch.argmax(logits_softmax[0, token_index, :]).numpy()
            max_id_value = logits_softmax[0, token_index, max_id].numpy()
            current_offsets = encoded.offset_mapping[0, token_index, :]
            word = sentence[current_offsets[0]:current_offsets[1]]
            tag_name = self.id_to_tag[max_id]
            entity_name = tag_name if tag_name == "O" else tag_name[2:]
            if (current_offsets[0] == 0 and current_offsets[1] == 0) or self.id_to_tag[max_id] == "O":
                if current_entity_word != "":
                    entities.append((EntityType(current_entity), current_entity_word))
                    current_entity_word = ""
                continue
            if tag_name[0] == "B":
                if current_entity_word != "":
                    entities.append((EntityType(current_entity), current_entity_word))
                current_entity = entity_name
                current_entity_word = word
            elif tag_name[0] == "I":
                current_entity_word += f" {word}"
        return entities
