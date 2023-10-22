from typing import List, Dict, Tuple
from src.data import Instance

B_PREF="B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
O = "O"

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"

import logging

logger = logging.getLogger(__name__)

def convert_iobes(tags):
    """
    Function to convert IOB tagging schema to IOBES.
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def build_label_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
	"""
	Build the mapping from label to index and index to labels.
	:param insts: list of instances.
	:return:
	"""
	label2idx = {}
	idx2labels = []
	label2idx[PAD] = len(label2idx)
	idx2labels.append(PAD)
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				idx2labels.append(label)
				label2idx[label] = len(label2idx)

	label2idx[START_TAG] = len(label2idx)
	idx2labels.append(START_TAG)
	label2idx[STOP_TAG] = len(label2idx)
	idx2labels.append(STOP_TAG)
	label_size = len(label2idx)
	logger.info("#labels: {}".format(label_size))
	logger.info("label 2idx: {}".format(label2idx))
	return idx2labels, label2idx

def check_all_labels_in_dict(insts: List[Instance], label2idx: Dict[str, int]):
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				raise ValueError(f"The label {label} does not exist in label2idx dict. The label might not appear in the training set.")


def build_word_idx(trains:List[Instance], devs:List[Instance], tests:List[Instance]) -> Tuple[Dict, List, Dict, List]:
	"""
	Build the vocab 2 idx for all instances
	:param train_insts:
	:param dev_insts:
	:param test_insts:
	:return:
	"""
	word2idx = dict()
	idx2word = []
	word2idx[PAD] = 0
	idx2word.append(PAD)
	word2idx[UNK] = 1
	idx2word.append(UNK)

	char2idx = {}
	idx2char = []
	char2idx[PAD] = 0
	idx2char.append(PAD)
	char2idx[UNK] = 1
	idx2char.append(UNK)

	# extract char on train, dev, test
	for inst in trains + devs + tests:
		for word in inst.words:
			if word not in word2idx:
				word2idx[word] = len(word2idx)
				idx2word.append(word)
	# extract char only on train (doesn't matter for dev and test)
	for inst in trains:
		for word in inst.words:
			for c in word:
				if c not in char2idx:
					char2idx[c] = len(idx2char)
					idx2char.append(c)
	return word2idx, idx2word, char2idx, idx2char


def check_all_obj_is_None(objs):
	for obj in objs:
		if obj is not None:
			return False
	return [None] * len(objs)
