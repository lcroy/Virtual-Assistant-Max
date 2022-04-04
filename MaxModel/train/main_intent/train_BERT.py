import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from configure import Config

from reliability_diagrams import *


def data_load(file_path, intent_label_file_path):
    # read all the data into pandas data
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['intent_label', 'sentence'])
    # mark the intent index based on the intents
    index = []
    temp_label = ''
    num = -1
    for _, row in df.iterrows():
        if str(row['intent_label']) != temp_label:
            num = num + 1
            temp_label = row['intent_label']
        index.append(num)
    # add a new column 'intent_index' to the data
    df['intent_index'] = index
    # generate the intent label text file
    df[['intent_label', 'intent_index']].drop_duplicates().to_csv(intent_label_file_path, sep='\t', index=False,
                                                                  header=False)
    # number of class
    num_cls = num + 1

    return df, num_cls


def create_example(df):
    # Create sentence and label lists
    sentences = df.sentence.values
    # Add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    # Get the intent index for label
    labels = df.intent_index.values

    # convert to token by using pre-trained the bert BERT_model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # obtain the tokens from the sentence list
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # define the maximum length of the sentence. The default value is 512. Here we use 64 because people won't say to
    # too much when he/she talk to robot
    MAX_LEN = 64
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens. If the sentence too long, then it will be cut, otherwise it will be padded with '0' to reach the length
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # MASK is required by BERT. BERT randomly mask 15% words of a sentence and predict what it is.
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Use train_test_split to split our data into train and validation sets for training
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=2018,
                                                                                        test_size=0.3)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                           random_state=2018, test_size=0.3)

    # Convert all of our data into torch tensors, the required datatype
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    return train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks

def plot_RD_CM(pred_y, true_y, confidence):

    plt.style.use("seaborn")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)

    fig = reliability_diagram(true_y, pred_y, confidence, num_bins=10, draw_ece=True,
                                  draw_bin_importance="alpha", draw_averages=True,
                                  title="reliability of prediction", figsize=(6, 6), dpi=100,
                                  return_fig=True)
    fig.savefig("RD.png",format="png", dpi=144, bbox_inches="tight", pad_inches=0.2)

    conf_matrix = confusion_matrix(pred_y, true_y)
    if conf_matrix.shape == (10, 10):
        df_cm = pd.DataFrame(conf_matrix, range(10), range(10))
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        plt.savefig("BERT_CM.jpg")
        plt.show()



# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train(train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks, num_cls,
          model_path):
    # Select a batch size for training. For fine-tuning BERT on a specific task, batch size of 16 or 32
    batch_size = 32
    # number of intents of the training data
    num_classes = num_cls
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because,
    # unlike a for loop, with an iterator the entire data does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Load BertForSequenceClassification, the pretrained BERT BERT_model with a single linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
    model.cuda()
    # set up optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        # Training

        # Set our BERT_model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # need to add this to windows/ linux does not has this problem
            b_input_ids = torch.tensor(b_input_ids).to(torch.int64)

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # Forward pass
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put BERT_model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        pred_y = np.array([])
        true_y = np.array([])
        confidence = np.array([])

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = torch.tensor(b_input_ids).to(torch.int64)

            # Telling the BERT_model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # for plotting
            pred_y = np.append(pred_y, np.argmax(logits, axis=1).flatten())
            true_y = np.append(true_y, label_ids.flatten())
            confidence_y = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            confidence_y = np.max(confidence_y, axis=-1)
            confidence = np.append(confidence, confidence_y)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        plot_RD_CM(pred_y, true_y, confidence)
    torch.save(model, model_path)


def predict(model_path, text, label_file_path):
    # the user's voice will be translated to text which will be used for predicting the user intent
    sentence = ["[CLS] " + text + " [SEP]"]
    # since we are not going to have a label for user text, we write some random number here
    label = [-100]

    # load the pre-trained bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # there is only one sentence here, that is sentence[0]
    tokenized_texts = [tokenizer.tokenize(sentence[0])]

    # define the maximum length of the sentence. The default value is 512. Here we use 64 because people won't say to
    # too much when he/she talk to robot
    MAX_LEN = 64
    # convert the tokens to ids for bert-training
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    # add the masks which is required by bert
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # convert them to tensor
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(label)

    # Select a batch size for training. For fine-tuning BERT on a specific task, batch size of 16 or 32
    batch_size = 32

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # load the trained BERT_model
    model = torch.load(model_path)

    model.eval()

    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(torch.int64)
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()

        predictions.append(logits)

    label_index = np.argmax(predictions[0], axis=1).flatten().item()

    df = pd.read_csv(label_file_path, delimiter='\t', header=None, names=['intent_label', 'intent_index'])

    label = df.loc[df['intent_index'] == label_index, ['intent_label']].values.item()

    # print(max(predictions[0][0]))

    return label, max(predictions[0][0])


def run_train():
    cfg = Config()

    # loading data and number of class
    data, num_cls = data_load(cfg.main_intent_dataset_csv, cfg.main_intent_label_csv)

    train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks = create_example(
        data)

    train(train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks, num_cls,
          cfg.main_model_BERT_path)

run_train()