#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

import chainer
import h5py
import logging
import numpy as np
import os
import random
import six
from tqdm import tqdm

from chainer.training import extension


def load_dataset(path, label_dict, outdir=None):
    """Load and save HDF5 that contains a dataset and stats for LM

    Args:
        path (str): The path of an input text dataset file
        label_dict (dict[str, int]):
            dictionary that maps token label string to its ID number
        outdir (str): The path of an output dir

    Returns:
        tuple[list[np.ndarray], int, int]: Tuple of
            token IDs in np.int32 converted by `read_tokens`
            the number of tokens by `count_tokens`,
            and the number of OOVs by `count_tokens`
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        filename = outdir + "/" + os.path.basename(path) + ".h5"
        if os.path.exists(filename):
            logging.info(f"loading binary dataset: {filename}")
            f = h5py.File(filename, "r")
            return f["data"][:], f["n_tokens"][()], f["n_oovs"][()]
    else:
        logging.info("skip dump/load HDF5 because the output dir is not specified")
    logging.info(f"reading text dataset: {path}")
    ret = read_tokens(path, label_dict)
    n_tokens, n_oovs = count_tokens(ret, label_dict["<unk>"])
    if outdir is not None:
        logging.info(f"saving binary dataset: {filename}")
        with h5py.File(filename, "w") as f:
            # http://docs.h5py.org/en/stable/special.html#arbitrary-vlen-data
            data = f.create_dataset(
                "data", (len(ret),), dtype=h5py.special_dtype(vlen=np.int32)
            )
            data[:] = ret
            f["n_tokens"] = n_tokens
            f["n_oovs"] = n_oovs
    return ret, n_tokens, n_oovs

def read_tokens_for_embedding(filename, spm_model, label_dict, subword2word_dict):
    import sentencepiece as spm
    import json
    data_load = json.load(open(filename))
    sp = spm.SentencePieceProcessor()
    sp.Init(model_file=spm_model)

    data = []
    unk = label_dict["<unk>"]
    label_all = np.array([])
    for word,sentce_list in tqdm(data_load.items()):
        for sentce in sentce_list:
            nt = 0  # number of token in this word
            subword_list=sp.encode(sentce[0], out_type=str)
            tmp = []; aver_mask = np.zeros([len(subword_list), len(subword_list)])
            for i in range(len(subword_list)):  # look thought a whole sentence
                label = subword_list[i]
                label_all = np.append(label_all, label_dict.get(label, unk))
                if "▁" in label:  # count "▁" from 1, 0 for <sos>.
                    tmp.append(nt)
                    if i > 0:
                        prev_pos = int(sum(tmp[0:len(tmp) - 1]))  # tmp[0,...,i-2]
                        curr_pos = int(sum(tmp[0:len(tmp)]))  # tmp[0,...,i-1]
                        aver_mask[prev_pos:curr_pos, prev_pos:curr_pos] = 1 / nt  # this is for previous word
                    nt = 0
                nt = nt + 1
            tmp.append(nt)
            prev_pos = int(sum(tmp[0:len(tmp) - 1]))  # tmp[0,...,i-2]
            curr_pos = int(sum(tmp[0:len(tmp)]))  # tmp[0,...,i-1]
            aver_mask[prev_pos:curr_pos, prev_pos:curr_pos] = 1 / nt

            # for word-level label
            tmp.remove(0); pos = 0; label_word_all = np.array([])
            for i in range(len(tmp)):
                label_in_word = label_all[pos:pos + tmp[i]].tolist()
                label_word = [subword2word_dict.get(tuple(label_in_word), unk)]
                padding = [val for val in [-100] for j in range(tmp[i] - 1)]
                label_word_multiple = label_word + padding
                label_word_all = np.append(label_word_all, label_word_multiple)
                pos = pos + tmp[i]
            if label_word_all.shape != label_all.shape:
                raise NotImplementedError("Something wrong with make batch.")
            word_index = sum(tmp[:sentce[1]])
            data.append([[np.array(label_all, dtype=np.int32),word_index], \
                         np.array(aver_mask), np.array(label_word_all, dtype=np.int32)])
            label_all = np.array([])
            label_word_all = np.array([])
            aver_mask = np.array([])
    return data
    # data=[[label_all, word_index], aver_mask, label_word_all]
    # word_index is the index of the first subword of the word that need to evaluate its representation

def read_tokens(filename, label_dict):
    """Read tokens as a sequence of sentences

    :param str filename : The name of the input file
    :param dict label_dict : dictionary that maps token label string to its ID number
    :return list of ID sequences
    :rtype list
    """

    data = []
    unk = label_dict["<unk>"]
    label_all = np.array([])
    for ln in tqdm(open(filename, "r", encoding="utf-8")):
        nt = 0 #number of token in this word
        tmp=[]; aver_mask = np.zeros([len(ln.split()), len(ln.split())])
        for i in range(len(ln.split())):
            label = ln.split()[i]
            label_all = np.append(label_all, label_dict.get(label, unk))
            if "▁" in label: #count "▁" from 1, 0 for <sos>.
                tmp.append(nt)
                if i > 0:
                    prev_pos=int(sum(tmp[0:len(tmp)-1]))#tmp[0,...,i-2]
                    curr_pos=int(sum(tmp[0:len(tmp)]))#tmp[0,...,i-1]
                    aver_mask[prev_pos:curr_pos,prev_pos:curr_pos]=1/nt #this is for previous word
                nt = 0
            nt = nt + 1
        tmp.append(nt)
        prev_pos = int(sum(tmp[0:len(tmp) - 1]))  # tmp[0,...,i-2]
        curr_pos = int(sum(tmp[0:len(tmp)]))  # tmp[0,...,i-1]
        aver_mask[prev_pos:curr_pos, prev_pos:curr_pos] = 1 / nt

        data.append([np.array(label_all, dtype=np.int32), np.array(aver_mask), tmp])
        label_all = np.array([])
        aver_mask = np.array([])
        # data.append(
        #     np.array(
        #         [label_dict.get(label, unk) for label in ln.split()], dtype=np.int32
        #     )
        # )
    return data


def count_tokens(data, unk_id=None):
    """Count tokens and oovs in token ID sequences.

    Args:
        data (list[np.ndarray]): list of token ID sequences
        unk_id (int): ID of unknown token

    Returns:
        tuple: tuple of number of token occurrences and number of oov tokens

    """

    n_tokens = 0
    n_oovs = 0
    for sentence in data:
        n_tokens += len(sentence[0])
        if unk_id is not None:
            n_oovs += np.count_nonzero(sentence[0] == unk_id)
    return n_tokens, n_oovs


def compute_perplexity(result):
    """Computes and add the perplexity to the LogReport

    :param dict result: The current observations
    """
    # Routine to rewrite the result dictionary of LogReport to add perplexity values
    result["perplexity"] = np.exp(result["main/loss"] / result["main/count"])
    if "validation/main/loss" in result:
        result["val_perplexity"] = np.exp(result["validation/main/loss"])


class ParallelSentenceIterator(chainer.dataset.Iterator):
    """Dataset iterator to create a batch of sentences.

    This iterator returns a pair of sentences, where one token is shifted
    between the sentences like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
    Sentence batches are made in order of longer sentences, and then
    randomly shuffled.
    """

    def __init__(
        self, dataset, batch_size, max_length=0, sos=0, eos=0, repeat=True, shuffle=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.batch_indices = []
        # make mini-batches
        if batch_size > 1:
            indices = sorted(range(len(dataset)), key=lambda i: -len(dataset[i][0]))
            bs = 0
            while bs < length:
                be = min(bs + batch_size, length)
                # batch size is automatically reduced if the sentence length
                # is larger than max_length
                if max_length > 0:
                    sent_length = len(dataset[indices[bs]][0])
                    be = min(
                        be, bs + max(batch_size // (sent_length // max_length + 1), 1)
                    )
                self.batch_indices.append(np.array(indices[bs:be]))
                bs = be
            if shuffle:
                # shuffle batches
                random.shuffle(self.batch_indices)
        else:
            self.batch_indices = [np.array([i]) for i in six.moves.range(length)]

        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        self.sos = sos
        self.eos = eos
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.0

    def __next__(self, aver_mask=None):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # represented by token IDs.
        n_batches = len(self.batch_indices)
        if not self.repeat and self.iteration >= n_batches:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration

        batch = []; batch_evlword_index = []
        aver_mask = []; tmp = []
        for idx in self.batch_indices[self.iteration % n_batches]:
            if isinstance(self.dataset[0][0], list):# only for evaluate word representation
                batch.append(
                    (
                        np.append([self.sos], self.dataset[idx][0][0]),
                        np.append(self.dataset[idx][0][0], [self.eos])
                    )
                )
                batch_evlword_index.append(self.dataset[idx][0][1])
            else:
                batch.append(
                    (
                        np.append([self.sos], self.dataset[idx][0]),
                        np.append(self.dataset[idx][0], [self.eos]),
                    )
                )
        if aver_mask is not None:
            for idx in self.batch_indices[self.iteration % n_batches]:
                add_1 = np.zeros([self.dataset[idx][1].shape[1],1])
                add_2 = np.zeros([1,self.dataset[idx][1].shape[1]+1])
                add_2[0][0]=1
                aver_mask.append(
                    (
                        np.vstack((add_2, np.hstack((add_1, self.dataset[idx][1]))))
                    )
                )
                tmp.append(self.dataset[idx][2])
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1

        epoch = self.iteration // n_batches
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        if aver_mask is not None:
            if isinstance(self.dataset[0][0], list):  # only for evaluate word representation
                return batch, aver_mask, batch_evlword_index
            return batch, aver_mask, tmp
        else:
            return batch

    def start_shuffle(self):
        random.shuffle(self.batch_indices)

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration / len(self.batch_indices)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer("iteration", self.iteration)
        self.epoch = serializer("epoch", self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                "previous_epoch_detail", self._previous_epoch_detail
            )
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + (
                self.current_position - 1
            ) / len(self.batch_indices)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(self._previous_epoch_detail, 0.0)
            else:
                self._previous_epoch_detail = -1.0


class MakeSymlinkToBestModel(extension.Extension):
    """Extension that makes a symbolic link to the best model

    :param str key: Key of value
    :param str prefix: Prefix of model files and link target
    :param str suffix: Suffix of link target
    """

    def __init__(self, key, prefix="model", suffix="best"):
        super(MakeSymlinkToBestModel, self).__init__()
        self.best_model = -1
        self.min_loss = 0.0
        self.key = key
        self.prefix = prefix
        self.suffix = suffix

    def __call__(self, trainer):
        observation = trainer.observation
        if self.key in observation:
            loss = observation[self.key]
            if self.best_model == -1 or loss < self.min_loss:
                self.min_loss = loss
                self.best_model = trainer.updater.epoch
                src = "%s.%d" % (self.prefix, self.best_model)
                dest = os.path.join(trainer.out, "%s.%s" % (self.prefix, self.suffix))
                if os.path.lexists(dest):
                    os.remove(dest)
                os.symlink(src, dest)
                logging.info("best model is " + src)

    def serialize(self, serializer):
        if isinstance(serializer, chainer.serializer.Serializer):
            serializer("_best_model", self.best_model)
            serializer("_min_loss", self.min_loss)
            serializer("_key", self.key)
            serializer("_prefix", self.prefix)
            serializer("_suffix", self.suffix)
        else:
            self.best_model = serializer("_best_model", -1)
            self.min_loss = serializer("_min_loss", 0.0)
            self.key = serializer("_key", "")
            self.prefix = serializer("_prefix", "model")
            self.suffix = serializer("_suffix", "best")


# TODO(Hori): currently it only works with character-word level LM.
#             need to consider any types of subwords-to-word mapping.
def make_lexical_tree(word_dict, subword_dict, word_unk):
    """Make a lexical tree to compute word-level probabilities"""
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            for i, c in enumerate(w):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root
