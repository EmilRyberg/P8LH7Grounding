import re
from transformers import DistilBertTokenizerFast
import torch
import numpy as np
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

def read_dataset(file_path):
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


class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def encode_tags(tags, encodings, tag_to_id):
    labels = [[tag_to_id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

def evaluate(dataset_path, model_weights_path, tag_path):
    with open(tag_path, "r") as tag_file:
        file_content = tag_file.read().strip()
    id_to_tag = file_content.splitlines()
    model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(id_to_tag))
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    texts, tags = read_dataset(dataset_path)
    unique_tags = file_content.splitlines()
    tag_to_id = { tag: id for id, tag in enumerate(unique_tags) }
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    labels = encode_tags(tags, encodings, tag_to_id)
    encodings.pop("offset_mapping") # we don't want to pass this to the model
    dataset = NERDataset(encodings, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)

    predictions = None
    labels = None
    for batch in dataloader:
        result = model(batch["input_ids"], batch["attention_mask"])
        if predictions is None:
            predictions = result.logits.detach().cpu()
            labels = batch["labels"].detach().cpu()
        else:
            predictions = torch.cat((predictions, result.logits.detach().cpu()), dim=0)
            labels = torch.cat((labels, batch["labels"].detach().cpu()), dim=0)

    predictions_softmax = torch.nn.Softmax(dim=2)(predictions)
    labels = labels.numpy()

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    total_predictions = 0
    for sentence_labels, sentence_predictions in zip(labels, predictions_softmax):
        max_ids = torch.argmax(sentence_predictions, dim=1).numpy()
        for label, prediction in zip(sentence_labels, max_ids):
            if label == -100:
                continue
            total_predictions += 1
            if label == prediction and label != 0: # TP
                tp += 1
            elif label != prediction and prediction != 0: # FP
                fp += 1
            elif label != prediction and prediction == 0 and label != 0: # FN
                fn += 1
            elif prediction == 0 and label == 0:
                tn += 1
            else:
                raise Exception("This should not happen, check your code")

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / total_predictions
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Test results:\n\n\tPrecision: {precision:.6f}\n\tRecall: {recall:.6f}\n\tF1 Score: {f1:.6f}\n\tAccuracy: {accuracy*100:.3f}%")


if __name__ == "__main__":
    evaluate("NER_5/test_dataset_processed.txt", "NER_5/pytorch_model.bin", "NER_5/tags.txt")
