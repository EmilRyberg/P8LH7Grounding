import re
import pathlib
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
import numpy as np
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments


def read_wnut(file_path):
    with open(file_path) as f:
        raw_text = f.read().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        tag_docs = []
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)

        return token_docs, tag_docs


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


texts, tags = read_wnut('output.txt')
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)
unique_tags = {
    "O": 0,
    "B-object": 1,
    "I-object": 2,
    "B-color": 3,
    "B-grasp": 4,
    "I-grasp": 5,
    "B-find": 6,
    "I-find": 7,
    "B-location": 8,
    "I-location": 9
} #set(tag for doc in tags for tag in doc)
print("unique tags", unique_tags)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
print("tag2id", tag2id)
id2tag = {id: tag for tag, id in tag2id.items()}
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
train_labels = encode_tags(train_tags, train_encodings)
print("train_labels", train_labels)
val_labels = encode_tags(val_tags, val_encodings)
print(val_texts[0], val_tags[0], sep='\n')
print(val_texts[1], val_tags[1], sep='\n')
train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))

training_args = TrainingArguments(
    output_dir='./results',          # output directory,
    overwrite_output_dir=True,
    num_train_epochs=100,            # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)
model.train()
trainer.train()
trainer.evaluate()
trainer.save_model("test")
print(val_encodings)
id_tensor = torch.tensor(val_encodings.input_ids).cuda()
mask = torch.tensor(val_encodings.attention_mask).cuda()
#print(id_tensor)
model.eval()
output = model(id_tensor, attention_mask=mask)
logits = output.logits

print(output)
