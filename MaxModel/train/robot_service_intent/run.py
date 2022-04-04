# The code is based on paper - JointBERT: BERT for Joint Intent Classification and Slot Filling
# The source code is modified based on huggingface and  https://github.com/monologg/JointBERT

import argparse
import logging
from transformers import BertTokenizer

from configure import Config
from trainer import Trainer
from data_loader import load_datasets


def main(args):
    # Have more information on what's happening under the hood, activate the logger as follows
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained BERT_model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # load data for training
    train_dataset = load_datasets(args, tokenizer, mode="train")
    dev_dataset = load_datasets(args, tokenizer, mode="dev")
    test_dataset = load_datasets(args, tokenizer, mode="test")
    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    # start to train BERT_model
    if args.do_train:
        trainer.train()
    # start to evaluate BERT_model
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    cfg = Config()
    # set up basic parameters
    parser.add_argument("--task", default='mir', type=str, help="Specifying the name of the task needs to be trained. The default one is webot.")
    parser.add_argument("--model_dir", default=cfg.mir_model_path, type=str, help="where we save the BERT_model.")
    parser.add_argument("--data_dir", default=cfg.dataset_path, type=str, help="the location of the data")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")
    parser.add_argument("--model_type", default="bert", type=str, help="Using Bert to achieve SOTA")
    parser.add_argument("--model_name_or_path", default = "bert-base-uncased", type=str, help="Using bert to achieve SOTA")

    parser.add_argument('--seed', type=int, default=10, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", default="True", help="Whether to run training.")
    parser.add_argument("--do_eval", default="True", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", default="False", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    args = parser.parse_args()

    main(args)
