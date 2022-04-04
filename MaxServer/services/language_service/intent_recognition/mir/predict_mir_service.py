import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer
from services.language_service.intent_recognition.mir.modeling_jointbert import JointBERT

class Predictor(object):

    def __init__(self, dataset_path, task, intent_label, slot_label):
        self.dataset_path = dataset_path
        self.intent_label = intent_label
        self.slot_label = slot_label

    # Obtain the intent label list
    def get_intent_labels(self):
        return [label.strip() for label in
                open(os.path.join(self.dataset_path, self.intent_label), 'r',
                     encoding='utf-8')]

    # obtain the slot label list
    def get_slot_labels(self):
        return [label.strip() for label in
                open(os.path.join(self.dataset_path, self.slot_label), 'r',
                     encoding='utf-8')]

    # obtain pre-defined the parameters when we train the model
    def get_args(self, pred_config):
        return torch.load(os.path.join(pred_config['model_dir'], 'training_args.bin'))

    # load the pre-trained model
    def load_model(self, model_dir, args, device):
        model = JointBERT.from_pretrained(model_dir, args=args, intent_label_lst=self.get_intent_labels(),
                                          slot_label_lst=self.get_slot_labels())
        model.to(device)
        model.eval()

        return model

    # read the input voice text
    def read_input_text(self, pred_config):
        lines = []
        line = pred_config['input_file'].strip()
        words = line.split()
        lines.append(words)

        return lines

    # convert the input text to feature for prediction
    def convert_input_file_to_tensor_dataset(self, lines,
                                             args,
                                             tokenizer,
                                             pad_token_label_id,
                                             cls_token_segment_id=0,
                                             pad_token_segment_id=0,
                                             sequence_a_segment_id=0,
                                             mask_padding_with_zero=True):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_slot_label_mask = []

        for words in lines:
            tokens = []
            slot_label_mask = []
            for word in words:
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [unk_token]  # For handling the bad-encoded word
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > args.max_seq_len - special_tokens_count:
                tokens = tokens[: (args.max_seq_len - special_tokens_count)]
                slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            slot_label_mask += [pad_token_label_id]

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids
            slot_label_mask = [pad_token_label_id] + slot_label_mask

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
            all_slot_label_mask.append(slot_label_mask)

        # Change to Tensor
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

        return dataset

    # predict the intent and slots
    def predict(self, pred_config):
        # load model and args
        args = self.get_args(pred_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        model = self.load_model(pred_config['model_dir'], args, device)

        intent_label_lst = self.get_intent_labels()
        slot_label_lst = self.get_slot_labels()

        # Convert input file to TensorDataset
        pad_token_label_id = args.ignore_index
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        lines = self.read_input_text(pred_config)
        dataset = self.convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)

        # Predict
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config['batch_size'])

        all_slot_label_mask = None
        intent_preds = None
        slot_preds = None

        for batch in tqdm(data_loader, desc="Predicting"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "intent_label_ids": None,
                          "slot_labels_ids": None}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                _, (intent_logits, slot_logits) = outputs[:2]

                # Intent Prediction
                if intent_preds is None:
                    intent_preds = intent_logits.detach().cpu().numpy()
                else:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

                # Slot prediction (keep the crf but not used it yet)
                if slot_preds is None:
                    if args.use_crf:
                        # decode() in `torchcrf` returns list with best index directly
                        slot_preds = np.array(model.crf.decode(slot_logits))
                    else:
                        slot_preds = slot_logits.detach().cpu().numpy()
                    all_slot_label_mask = batch[3].detach().cpu().numpy()
                else:
                    if args.use_crf:
                        slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
                    else:
                        slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                    all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

        # 1. the most likely intent ids
        confidence = np.max(intent_preds[0])
        intent_preds = np.argmax(intent_preds, axis=1)

        if args.use_crf:
            slot_preds = np.array(model.crf.decode(slot_logits))
        else:
            slot_preds = np.argmax(slot_preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
        slot_preds_list = [[] for _ in range(slot_preds.shape[0])]


        for i in range(slot_preds.shape[0]):
            for j in range(slot_preds.shape[1]):
                if all_slot_label_mask[i, j] != pad_token_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        # 2. zip the entities + slot labels
        slot_list = []
        for words, slot_preds in zip(lines[0], slot_preds_list[0]):
            if slot_preds != 'O':
                temp = []
                temp = [words, slot_preds]
                slot_list.append(temp)

        # 3. final results: dict("intent": xxxx, "slot": [[entity_one, slot_label_one], [entity_two, slot_label_two]...])
        pred_results = dict(intent=intent_label_lst[intent_preds[0]], slot=slot_list)

        # # Write to output file
        # with open(pred_config['output_file'], "w", encoding="utf-8") as f:
        #     for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
        #         line = ""
        #         for word, pred in zip(words, slot_preds):
        #             if pred == 'O':
        #                 line = line + word + " "
        #             else:
        #                 line = line + "[{}:{}] ".format(word, pred)
        #         f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

        return pred_results, confidence
#
def run_pred_mir(cfg, input_text, batch_size):
    pred = Predictor(cfg.mir_dataset_path, cfg.mir_task, cfg.mir_intent_label, cfg.mir_slot_label)

    pred_config = dict(config=cfg, input_file=input_text, model_dir=cfg.mir_model_path,
                       batch_size=batch_size)

    pred_result, confidence = pred.predict(pred_config)
    # print(confidence)
    print(pred_result, confidence)
    return pred_result, confidence
