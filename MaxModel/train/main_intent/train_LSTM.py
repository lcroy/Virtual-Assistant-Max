# ==============================================================
# LSTM Model for Main intent recognition
# Author: Chen Li
# Description: Identify the requested service, e.g., MiR, other
# Date: 21/02/2021
#===============================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import spacy
import re
import string
import json
import seaborn as sn

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from configure import Config
from torch.nn.utils.rnn import pack_padded_sequence


cfg = Config()

# load the data from the data file
def data_load():
    # read csv data to dataframe
    org_df = pd.read_csv(cfg.main_intent_dataset_csv, sep='\t', header=None, names=['label', 'text'])
    #swich the columns
    new_df = org_df[['text','label']]

    # read label data to dict
    dict = {row[0]: row[1] for _, row in pd.read_csv(cfg.main_intent_label_csv, sep='\t', header=None).iterrows()}
    # assign the index to the label for each intent
    new_df['label'] = [dict.get(label) for label in new_df['label']]

    return new_df

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

# Training the model
def train_model(model, optimizer, criterion, train_loader, valid_loader, epochs):
    # for plotting the results
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []

    #Start to train
    for i in range(epochs):
        model.train()
        # for calculating the loss and accuracy
        sum_loss = 0.0
        total = 0
        train_acc = 0

        for x, y, l in train_loader:
            x = x.long()
            y = y.long()
            # pytorch will auto-add grad but you need to clear grad manually
            optimizer.zero_grad()
            # calculate model output
            y_pred = model(x, l)
            # calculate loss
            loss = criterion(y_pred, y)
            # calculate grad
            loss.backward()
            # update weight
            optimizer.step()
            # output loss
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
            # output accuracy
            pred = torch.max(y_pred, 1)[1]
            train_acc += (pred == y).float().sum()
            # train_acc += (y_pred.argmax(1) == y).sum().item()

        # validate the model
        val_loss, val_acc, val_rmse = evaluate_model(model, criterion, valid_loader)

        if i % 2 == 0:
            print("train loss %.3f, train accuracy %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, train_acc/total, val_loss, val_acc, val_rmse))

        #accumulate the training/validation loss and accuracy
        train_loss.append(sum_loss/total)
        valid_loss.append(val_loss)
        train_accuracy.append(train_acc/total)
        valid_accuracy.append(val_acc)

    # plot the loss and accuracy
    plot_loss_acc(train_loss, valid_loss, train_accuracy, valid_accuracy)

    # save the model state
    torch.save(model.state_dict(), cfg.main_model_LSTM_path)

# evaluation model
def evaluate_model(model, criterion, valid_loader):
    # for calculating accuracy and loss
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0

    # eval() model, close dropout and batch normalization
    model.eval()
    with torch.no_grad():
        for x, y, l in valid_loader:
            x = x.long()
            y = y.long()
            y_hat = model(x, l)
            # calculate loss
            loss = criterion(y_hat, y)
            # output accuracy
            pred = torch.max(y_hat, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            # output loss
            sum_loss += loss.item()*y.shape[0]
            sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
        # print the report + confusion matrix
        print(classification_report(y,pred))
        conf_matrix = confusion_matrix(pred, y)
        print(conf_matrix)
        df_cm = pd.DataFrame(conf_matrix, index=["greeting","who_made_you","repeat", "yes", "how_old_are_you", "tell_joke", "are_you_a_bot", "play_music", "what_are_your_hobbies", "goodbye"], columns=["greeting","who_made_you","repeat", "yes", "how_old_are_you", "tell_joke", "are_you_a_bot", "play_music", "what_are_your_hobbies", "goodbye"])
        plt.figure(figsize=(15,15))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        plt.savefig('CM.jpg')
        plt.show()

    return sum_loss/total, correct/total, sum_rmse/total

# plot loss and accuracy
def plot_loss_acc(train_loss, valid_loss, train_accuracy, valid_accuracy):
    plt.plot(train_loss, label="Training loss")
    plt.plot(valid_loss, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig('Losses.jpg')
    plt.show()

    plt.plot(train_accuracy, label="Training accuracy")
    plt.plot(valid_accuracy, label="Validation accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig('Accuracy.jpg')
    plt.show()


if __name__ == "__main__":
    #load data
    df = data_load()

    # tockenize
    token = spacy.load('en_core_web_sm')
    counts = Counter()
    for _, item in df.iterrows():
        counts.update(tokenize(token, item['text']))

    # create vocab index
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    # size of the dictionary of embeddings
    num_embeddings  = len(words)

    # write the vocab2index and size of the dictionary of embeddings to a json file.
    with open(cfg.vocab_json, 'w') as outfile:
        data = {}
        data['vocab2index'] = vocab2index
        data['num_embeddings'] = num_embeddings
        json.dump(data, outfile)
        outfile.close()

    # encode the original text
    df['encoded'] = df['text'].apply(lambda x: np.array(encode_sentence(token, x, vocab2index, cfg.input_length)))
    X = list(df['encoded'])
    y = list(df['label'])

    # split the dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)

    # create Tensor datasets
    train_ds = CreateDataset(X_train, y_train)
    valid_ds = CreateDataset(X_valid, y_valid)

    # shuffle the data
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size)

    # model parameters

    # create a model
    model = LSTM_Main_Intent(num_embeddings, cfg.embedding_dim, cfg.hidden_dim, cfg.droupout, cfg.num_class)
    # set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    #train model
    train_model(model, optimizer, criterion, train_loader, valid_loader, epochs=cfg.epochs)
# ================================================================================================================
#     text_to_be_tested = {'label': [0], 'text': ["Sorry, I did not catch that" ]}
#     df_test = pd.DataFrame(data=text_to_be_tested)
#     df_test['encoded'] = df_test['text'].apply(lambda x: np.array(encode_sentence(tok, x, vocab2index, cfg.input_length)))
#     X = list(df_test['encoded'])
#     y = list(df_test['label'])
#     # print(df_test)
#     #
#     # # text_to_be_tested = {'how was your day', 0}
#     # # df_test = pd.DataFrame(data=text_to_be_tested)
#     # #
#     # # text_to_be_tested = "how was your day"
#     #
#     # # text_encode = np.array(encode_sentence(tok, text_to_be_tested, vocab2index))
#     #
#     test_data = CreateDataset(X, y)
#     test_loader = DataLoader(test_data, batch_size=1)
#
#     model_2 = LSTM_Main_Intent(input_size, cfg.embedding_dim, cfg.hidden_dim, cfg.droupout, cfg.num_class)
#     model_2.load_state_dict(torch.load(cfg.main_model_path))
#     model_2.eval()
#     for x, y, l in test_loader:
#         x = x.long()
#         # y = y.long()
#         y_hat = model_2(x, l)
#         pred = torch.max(y_hat, 1)[1]
#         print(pred)
#
#
#
#
#
