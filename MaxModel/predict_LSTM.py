import numpy as np
import pandas as pd
import torch
import spacy
import json
import time

from train.main_intent.train_LSTM import CreateDataset
from train.main_intent.train_LSTM import LSTM_Main_Intent
from train.main_intent.train_LSTM import encode_sentence
from torch.utils.data import DataLoader
from configure import Config

cfg = Config()

def pred_intent(text):
    token = spacy.load('en_core_web_sm')
    text_to_be_tested = {'label': [0], 'text': [text]}
    df_test = pd.DataFrame(data=text_to_be_tested)
    # read the vocab2index and size of the dictionary of embeddings from a json file.
    with open(cfg.vocab_json, 'r') as file:
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
    model_2.load_state_dict(torch.load(cfg.main_model_LSTM_path))
    # predict model
    model_2.eval()
    for x, y, l in test_loader:
        x = x.long()
        y_hat = model_2(x, l)
        input_max, input_indexes = torch.max(y_hat, 1)
        # print(input_max.item())
        pred_index = input_indexes

    label_dict = {row[0]: row[1] for _, row in pd.read_csv(cfg.main_intent_label_csv, sep='\t', header=None, names=['label', 'index']).iterrows()}
    pred_label = [label for label, index in label_dict.items() if index == pred_index]

    print("The predict intent is " + pred_label[0] + " and the confidence is " + str(input_max.item()))

if __name__ == "__main__":
    start_time = time.time()
    print("Response time for LSTM model:")
    print("Intent - Greeting")
    pred_intent('Hi max, how are you today')
    print("---The LSTM model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - who_made_you")
    pred_intent('Hi max, who made you?')
    print("---The LSTM model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - repeat")
    pred_intent('Hi max, can you say it again')
    print("---The LSTM model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - how_old_are_you")
    pred_intent('Hi max, how old are you')
    print("---The LSTM model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - tell_joke")
    pred_intent('Hi max, can you tell me a joke')
    print("---The LSTM model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - are_you_a_bot")
    pred_intent('Hi max, what are you')
    print("---The LSTM model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - what_are_your_hobbies")
    pred_intent('Hi max, what do you do in your free time')
    print("---The LSTM model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))

    start_time = time.time()
    print("Intent - goodbye")
    pred_intent('Hi max, see you later')
    print("---The LSTM model costs {:.3f} seconds to predict the final results---".format((time.time() - start_time)))