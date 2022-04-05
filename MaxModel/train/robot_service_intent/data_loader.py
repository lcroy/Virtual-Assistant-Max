import os
import logging
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a_input, text_b_slot_label=None, intent_label=None):
        """Constructs a InputExample

        Args:
            guid: Unique id for the example.
            text_a_input: list. The words of the sequence.
            text_b_slot_labels: (Optional) string. The webot label of the example.
            intent_label: (Optional) list. The slot labels of the example.
        """
        self.guid = guid
        self.text_a_input = text_a_input
        self.text_b_slot_label = text_b_slot_label
        self.intent_label = intent_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids


class mirProcessor(object):
    """Processor for the webot data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = self._get_intent_labels(args)
        self.slot_labels = self._get_slot_labels(args)
        # assign the input text file, label file, and slot label file
        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_label_file = 'seq.out'

    # obtain the intent label from pre-defined txt file
    @classmethod
    def _get_intent_labels(self, args):
        return [label.strip() for label in
                open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]

    # obtain the slot label from pre-defined txt file
    @classmethod
    def _get_slot_labels(self, args):
        return [label.strip() for label in
                open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]

    def get_example(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_example(input_text_file=self._read_file(os.path.join(data_path, self.input_text_file)),
                                    intent_lable_file=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                    slot_label_file=self._read_file(os.path.join(data_path, self.slot_label_file)),
                                    set_type=mode)

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_example(self, input_text_file, intent_lable_file, slot_label_file, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (input_text, intent_label, slot_label) in enumerate(
                zip(input_text_file, intent_lable_file, slot_label_file)):
            guid = "%s-%s" % (set_type, i)
            temp_input_text = input_text.split()  # Some are spaced twice
            temp_intent_label = self.intent_labels.index(
                intent_label) if intent_label in self.intent_labels else self.intent_labels.index("UNK")
            slot_labels = []
            for s in slot_label.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(temp_input_text) == len(slot_labels)
            examples.append(InputExample(guid=guid, text_a_input=temp_input_text, text_b_slot_label=slot_labels,
                                         intent_label=temp_intent_label))
        return examples

class cozmoProcessor(object):
    """Processor for the webot data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = self._get_intent_labels(args)
        self.slot_labels = self._get_slot_labels(args)
        # assign the input text file, label file, and slot label file
        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_label_file = 'seq.out'

    # obtain the intent label from pre-defined txt file
    @classmethod
    def _get_intent_labels(self, args):
        return [label.strip() for label in
                open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]

    # obtain the slot label from pre-defined txt file
    @classmethod
    def _get_slot_labels(self, args):
        return [label.strip() for label in
                open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]

    def get_example(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        print(data_path)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_example(input_text_file=self._read_file(os.path.join(data_path, self.input_text_file)),
                                    intent_lable_file=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                    slot_label_file=self._read_file(os.path.join(data_path, self.slot_label_file)),
                                    set_type=mode)

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_example(self, input_text_file, intent_lable_file, slot_label_file, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (input_text, intent_label, slot_label) in enumerate(
                zip(input_text_file, intent_lable_file, slot_label_file)):
            guid = "%s-%s" % (set_type, i)
            temp_input_text = input_text.split()  # Some are spaced twice
            temp_intent_label = self.intent_labels.index(
                intent_label) if intent_label in self.intent_labels else self.intent_labels.index("UNK")
            slot_labels = []
            for s in slot_label.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))
            assert len(temp_input_text) == len(slot_labels)

            examples.append(InputExample(guid=guid, text_a_input=temp_input_text, text_b_slot_label=slot_labels,
                                         intent_label=temp_intent_label))
        return examples

processors = {
    "mir": mirProcessor,
    "cozmo": cozmoProcessor
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for input_text, slot_label in zip(example.text_a_input, example.text_b_slot_label):
            word_tokens = tokenizer.tokenize(input_text)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_datasets(args, tokenizer, mode):
    # get the processors
    processor = processors[args.task](args)

    # Load data features from data file
    logger.info("Creating features from data file at %s", args.data_dir)

    if mode == "train":
        examples = processor.get_example("train")
    elif mode == "dev":
        examples = processor.get_example("dev")
    elif mode == "test":
        examples = processor.get_example("test")
    else:
        raise Exception("You provide some unknown datasets.")

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = args.ignore_index
    features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                            pad_token_label_id=pad_token_label_id)

    # Convert to Tensors and build data
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
    return dataset
