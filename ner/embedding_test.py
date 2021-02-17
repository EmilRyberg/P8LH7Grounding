import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import numpy as np
from scipy.spatial.distance import cosine

def test_model():
    model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=4)
    #model.load_state_dict(torch.load(model_path))
    model.eval()
    #print(model.get_input_embeddings())
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    run = True
    while run:
        print("Input sentence to test:")
        text = input("> ")
        encoded = tokenizer(text, is_split_into_words=False, return_offsets_mapping=True, padding=True,
                                    truncation=True, return_tensors="pt")
        print(encoded.input_ids.shape)
        print(tokenizer.decode(encoded.input_ids[0]))
        #embeddings = model.get_input_embeddings()
        #emb = embeddings(encoded.input_ids).cpu().detach().numpy()[0, 1:-1, :]
        #print(emb)
        output = model(encoded.input_ids, encoded.attention_mask, output_hidden_states=True)
        print(output.hidden_states[0].shape)
        hidden_state = output.hidden_states[-2].cpu().detach().numpy()[-1, :, :]
        print(hidden_state.shape)
        hidden_state_1_avg = np.mean(hidden_state, 0)
        print(hidden_state_1_avg.shape)
        if hidden_state_1_avg.ndim == 1:
            hidden_state_1_avg = np.expand_dims(hidden_state_1_avg, 1)
        #print(hidden_state_1_avg.shape)

        print("Input sentence 2 to test:")
        text = input("> ")
        encoded = tokenizer(text, is_split_into_words=False, return_offsets_mapping=True, padding=True,
                                    truncation=True, return_tensors="pt")
        print(encoded.input_ids.shape)
        print(tokenizer.decode(encoded.input_ids[0]))
        #emb = embeddings(encoded.input_ids).cpu().detach().numpy()[0, 1:-1, :]
        # print(emb)
        output = model(encoded.input_ids, encoded.attention_mask, output_hidden_states=True)
        hidden_state = output.hidden_states[-2].cpu().detach().numpy()[-1, :, :]
        hidden_state_2_avg = np.mean(hidden_state, 0)
        print(hidden_state_2_avg)
        if hidden_state_2_avg.ndim == 1:
            hidden_state_2_avg = np.expand_dims(hidden_state_2_avg, 1)
        print(f"dot: {np.dot(hidden_state_1_avg.transpose(), hidden_state_2_avg)}")
        cosine_sim = 1 - cosine(hidden_state_1_avg, hidden_state_2_avg) #np.dot(hidden_state_1_avg.transpose(), hidden_state_2_avg) / (np.linalg.norm(hidden_state_1_avg) * np.linalg.norm(hidden_state_2_avg))
        print(f"Cosine similarity: {cosine_sim}")

if __name__ == "__main__":
    test_model()