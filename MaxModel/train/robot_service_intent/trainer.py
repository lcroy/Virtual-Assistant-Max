import os
import logging
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from BERT_model.modeling_jointbert import JointBERT
from utils import compute_metrics
from reliability_diagrams import *
from sklearn.preprocessing import MultiLabelBinarizer

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.logger = self.set_loger()
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        # Obtain the intent and slot labels
        self.intent_label_lst = self._get_intent_labels(args)
        self.slot_label_lst =self._get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = (BertConfig, JointBERT, BertTokenizer)
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)

        # GPU or CPU
        # self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.device = "cpu"
        self.model.to(self.device)

    def set_loger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        # create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handler to the logger
        logger.addHandler(handler)

        return logger


    # obtain the intent label from pre-defined txt file
    def _get_intent_labels(self, args):
        return [label.strip() for label in
                open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]

    # obtain the slot label from pre-defined txt file
    def _get_slot_labels(self, args):
        return [label.strip() for label in
                open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                        len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(self.train_dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Total train batch size = %d", self.args.train_batch_size)
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)
        self.logger.info("  Logging steps = %d", self.args.logging_steps)
        self.logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                # if self.args.model_type != 'distilbert':
                #     inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        self.evaluate("dev")

        return global_step, tr_loss / global_step

    def evaluate(self, mode):

        confidence = np.array([])

        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test data available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        self.logger.info("***** Running evaluation on %s data *****", mode)
        self.logger.info("  Num examples = %d", len(dataset))
        self.logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(),
                                                axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        confidence_y = np.exp(intent_preds) / np.sum(np.exp(intent_preds), axis=1, keepdims=True)
        confidence_y = np.max(confidence_y, axis=-1)
        confidence = np.append(confidence, confidence_y)

        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)

        # add intent
        self.plot_RD_CM(intent_preds, out_intent_label_ids, confidence)

        # pred_slot = MultiLabelBinarizer().fit_transform(slot_preds_list)
        # real_slot = MultiLabelBinarizer().fit_transform(out_slot_label_list)
        # conf_matrix = confusion_matrix(pred_slot, real_slot)
        # df_cm = pd.DataFrame(conf_matrix, range(16), range(16))
        # plt.figure(figsize=(10, 7))
        # sn.set(font_scale=1.4)  # for label size
        # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        # plt.savefig("MiR_CM_sl.jpg")
        # plt.show()

        results.update(total_result)

        self.logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            self.logger.info("  %s = %s", key, str(results[key]))

        return results

    def plot_RD_CM(self, pred_y, true_y, confidence):

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
        fig.savefig("RD.png", format="png", dpi=144, bbox_inches="tight", pad_inches=0.2)

        conf_matrix = confusion_matrix(pred_y, true_y)
        if conf_matrix.shape == (8, 8):
            df_cm = pd.DataFrame(conf_matrix,
                                 index=["GREETING", "MISSIONCHECK", "POSITIONCHECK", "BATTERYCHECK", "ASKHELP",
                                        "STATESTOP", "STATERUN", "DELIVERY"], columns=["GREETING", "MISSIONCHECK",
                                                                                       "POSITIONCHECK", "BATTERYCHECK",
                                                                                       "ASKHELP", "STATESTOP",
                                                                                       "STATERUN", "DELIVERY"])

            # Normalise
            df_cm = df_cm.astype('float')/df_cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(15, 15))
            plt.xticks(rotation=45)
            plt.yticks(rotation=90)
            sn.set(font_scale=1.4)  # for label size
            sn.heatmap(df_cm, annot=True, fmt='.2f')  # font size
            plt.savefig("MiR_CM.jpg")
            plt.show(block=False)

    def save_model(self):
        # Save BERT_model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained BERT_model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        self.logger.info("Saving BERT_model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether BERT_model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            self.logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some BERT_model files might be missing...")
