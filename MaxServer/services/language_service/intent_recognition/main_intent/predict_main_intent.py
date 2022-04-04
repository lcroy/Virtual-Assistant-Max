import numpy as np
import pandas as pd
import torch
import spacy
import json
import string
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
from configure import Config

#dataset
class CreateDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

#LSTM Model
class LSTM_Main_Intent(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, droupout, num_class):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(droupout)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_class)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out

# Tokenization
# removing punctuation, special characters, and lower casing
def tokenize (tok, text):
    #reg
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text.lower())

    return [token.text for token in tok.tokenizer(nopunct)]

# Encode the sentence
def encode_sentence(tok, text, vocab2index, input_length):
    tokenized = tokenize(tok, text)
    encoded = np.zeros(input_length, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(input_length, len(enc1))
    encoded[:length] = enc1[:length]

    return encoded, length

# predict
def pred_intent(cfg, text):
    token = spacy.load('en_core_web_sm')
    text_to_be_tested = {'label': [0], 'text': [text]}
    df_test = pd.DataFrame(data=text_to_be_tested)
    # read the vocab2index and size of the dictionary of embeddings from a json file.
    with open(cfg.main_intent_vocab_json, 'r') as file:
        data = json.load(file)
        vocab2index = data['vocab2index']
        num_embeddings = data['num_embeddings']
        file.close()
    # encode text
    df_test['encoded'] = df_test['text'].apply(lambda x: np.array(encode_sentence(token, x, vocab2index, cfg.input_length)))
    X = list(df_test['encoded'])
    y = list(df_test['label'])
    # format text
    test_data = CreateDataset(X, y)
    test_loader = DataLoader(test_data, batch_size=1)
    # initial model
    model_2 = LSTM_Main_Intent(num_embeddings, cfg.embedding_dim, cfg.hidden_dim, cfg.droupout, cfg.num_class)
    # load trained model
    model_2.load_state_dict(torch.load(cfg.main_intent_model_LSTM_path))
    # predict model
    model_2.eval()
    for x, y, l in test_loader:
        x = x.long()
        y_hat = model_2(x, l)
        input_max, input_indexes = torch.max(y_hat, 1)
        # print(input_max.item())
        pred_index = input_indexes

    label_dict = {row[0]: row[1] for _, row in pd.read_csv(cfg.main_intent_label_csv_path, sep='\t', header=None, names=['label', 'index']).iterrows()}
    pred_label = [label for label, index in label_dict.items() if index == pred_index]

    print("The predict intent is " + pred_label[0] + " and the confidence is " + str(input_max.item()))
    return pred_label[0], input_max.item()