import os
import pickle
import random
import numpy as np

from data.data_utils import InputExamples
from nltk.tag.perceptron import PerceptronTagger
from allennlp.modules.elmo import batch_to_ids

from models.bert import tokenization_bert
from models.roberta import tokenization_roberta
from models.xlnet import tokenization_xlnet

# Code is widely inspired from:
# https://github.com/google-research/bert
class MetonymyProcessor(object):
  def __init__(self, hparams, dataset_type="relocar"):
    self.hparams = hparams
    self.dataset_type = dataset_type

    self._get_labels()
    self._get_word_dict()
    self._pad_idx = self.hparams.pad_idx

    # POS Tagger
    perceptron_tagger = PerceptronTagger()
    self.pos2id={"<PAD>" : 0}
    for pos_tag in list(perceptron_tagger.classes):
      self.pos2id[pos_tag] = len(self.pos2id)
    # Stanford NER
    self.ne2id = {"<PAD>": 0, "O": 1, "LOCATION": 2, "ORGANIZATION": 3, "PERSON": 4}

    if self.hparams.do_bert and not self.hparams.do_roberta and not self.hparams.do_xlnet:
      self._bert_tokenizer_init()
    elif self.hparams.do_bert and self.hparams.do_roberta:
      self._roberta_tokenizer_init()
    elif self.hparams.do_bert and self.hparams.do_xlnet:
      self._xlnet_tokenizer_init()

  def _bert_tokenizer_init(self, bert_pretrained='bert-base-cased'):
    bert_pretrained_dir = "/mnt/raid5/shared/bert/pytorch/%s/" % bert_pretrained
    vocab_file_path = "%s-vocab.txt" % bert_pretrained

    self._tokenizer = tokenization_bert.BertTokenizer(
      vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path),
      do_lower_case=False
    )
    print("bert_tokenizer")

  def _xlnet_tokenizer_init(self, xlnet_pretrained='xlnet-base-cased'):
    xlnet_pretrained_dir = "/mnt/raid5/shared/bert/pytorch/%s/" % xlnet_pretrained
    vocab_file_path = "%s-spiece.model" % xlnet_pretrained

    self._tokenizer = tokenization_xlnet.XLNetTokenizer(
      vocab_file=os.path.join(xlnet_pretrained_dir, vocab_file_path),
      do_lower_case=False
    )
    print("xlnet_tokenizer")

  def _roberta_tokenizer_init(self, roberta_pretrained='roberta-base'):
    roberta_pretrained_dir = "/mnt/raid5/shared/bert/pytorch/%s/" % roberta_pretrained
    vocab_file_path = "%s-vocab.json" % roberta_pretrained
    merges_file_path = "%s-merges.txt" % roberta_pretrained

    self._tokenizer = tokenization_roberta.GPT2Tokenizer(
      vocab_file=os.path.join(roberta_pretrained_dir, vocab_file_path),
      merges_file=os.path.join(roberta_pretrained_dir, merges_file_path)
    )
    print("roberta_tokenizer")

  # data_dir
  def get_train_examples(self, data_dir):
    self.train_example = self._read_pkl(os.path.join(data_dir % self.dataset_type,
                                                     self.hparams.train_pkl % self.dataset_type), do_shuffle=True)

    return self.train_example

  def get_relocar_test_examples(self):
    print("RELOCAR Test Set")
    self.relocar_test_example = self._read_pkl(os.path.join("data/relocar/%s" % self.hparams.relocar_test_pkl),
                                               do_shuffle=False)
    return self.relocar_test_example

  def get_semeval_test_examples(self):
    print("SEMEVAL Test Set")
    self.semeval_test_example = self._read_pkl(os.path.join("data/semeval/%s" % self.hparams.semeval_test_pkl),
                                               do_shuffle=False)
    return self.semeval_test_example

  def _get_labels(self):
    """See base class."""
    self.id2label = ["<PAD>", "<MET>", "<LIT>", "O"]
    self.label2id = dict()
    for idx, label in enumerate(self.id2label):
      self.label2id[label] = idx

  def _read_pkl(self, data_dir, do_shuffle=False):
    print("[Reading %s]" % data_dir)
    with open(data_dir, "rb") as frb_handle:
      total_examples = pickle.load(frb_handle)

      if do_shuffle and self.hparams.training_shuffle_num > 1:
        total_examples = self._data_shuffling(total_examples, self.hparams.training_shuffle_num)

      return total_examples

  def _data_shuffling(self, inputs, shuffle_num):
    for i in range(shuffle_num):
      random_seed = random.sample(list(range(0, 1000)), 1)[0]
      random.seed(random_seed)
      random.shuffle(inputs)
    # print("Shuffling total %d process is done! Total dialog context : %d" % (shuffle_num, len(inputs)))

    return inputs

  def _get_word_dict(self):
    with open(self.hparams.glove_vocab_dir % (self.dataset_type, self.dataset_type)) as vocab_handle:
      self.vocab = [word.strip() for word in vocab_handle if len(word.strip()) > 0]

    self.word2id = dict()
    for idx, word in enumerate(self.vocab):
      self.word2id[word] = idx

  def get_batch_data(self, curr_index, batch_size, set_type="train"):
    inputs_id = []
    labels_id = []
    sequence_lengths = []
    target_entities = []
    entities_idx = []
    inputs_pos = []
    inputs_ner = []

    examples = {
      "train": self.train_example,
      "relocar_test": self.relocar_test_example,
      "semeval_test": self.semeval_test_example,
    }
    example = examples[set_type]

    for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):

      input_id, label_id, input_pos, input_ner = \
        convert_single_example(each_example, label2id=self.label2id, word2id=self.word2id,
                               do_pos=True, pos2id=self.pos2id, do_ner=True, ne2id=self.ne2id)

      if self.hparams.do_use_elmo:
        input_id = each_example.inputs
        for idx, word_tok in enumerate(input_id):
          if not word_tok in self.word2id.keys():
            input_id[idx] = "<UNK>"

      inputs_id.append(input_id)
      labels_id.append(label_id)
      sequence_lengths.append(each_example.seq_len)
      target_entities.append(each_example.target_entity)
      entities_idx.append(each_example.entity_idx)
      inputs_pos.append(input_pos)
      inputs_ner.append(input_ner)

    if self.hparams.do_use_elmo:
      #char inputs [batch, max_seq_len, max_word_len(50)]
      pad_inputs_id = batch_to_ids(inputs_id)
    else:
      pad_inputs_id = rank_2_pad_process(inputs_id)
    pad_labels_id = rank_2_pad_process(labels_id)
    pad_inputs_pos = rank_2_pad_process(inputs_pos)
    pad_inputs_ner = rank_2_pad_process(inputs_ner)

    return pad_inputs_id, pad_labels_id, pad_inputs_pos, pad_inputs_ner, \
           sequence_lengths, target_entities, entities_idx

  def get_bert_batch_data(self, curr_index, batch_size, set_type="train"):
    inputs_id = []
    labels_id = []
    sequence_lengths = []
    target_entities = []
    entities_idx = []

    examples = {
      "train": self.train_example,
      "relocar_test": self.relocar_test_example,
      "semeval_test": self.semeval_test_example,
    }
    example = examples[set_type]

    for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):

      input_id = self._tokenizer.convert_tokens_to_ids(each_example.inputs)
      label_id = [self.label2id[label] for label in each_example.labels]

      inputs_id.append(input_id)
      labels_id.append(label_id)
      sequence_lengths.append(each_example.seq_len)
      target_entities.append(each_example.target_entity)
      entities_idx.append(each_example.entity_idx) # (start, end)

    pad_inputs_id = rank_2_pad_process(inputs_id, self._pad_idx)
    pad_labels_id = rank_2_pad_process(labels_id, self._pad_idx)

    return pad_inputs_id, pad_labels_id, sequence_lengths, target_entities, entities_idx

  def get_word_embeddings(self):
    with np.load(self.hparams.glove_embedding_path % (self.dataset_type, self.dataset_type)) as data:
      print("glove embedding shape", np.shape(data["embeddings"]))
      return data["embeddings"]

def rank_2_pad_process(inputs, pad_idx=0):

  max_sent_len = 0
  for sent in inputs:
    max_sent_len = max(len(sent), max_sent_len)

  padded_result = []
  sent_buffer = []
  for sent in inputs:
    for i in range(max_sent_len - len(sent)):
      sent_buffer.append(pad_idx)
    sent.extend(sent_buffer)
    padded_result.append(sent)
    sent_buffer = []

  return padded_result

def convert_single_example(example:InputExamples, label2id, word2id,
                           do_pos=True, pos2id=None, do_ner=True, ne2id=None):

  input_pos, input_ner = None, None
  if do_pos:
    input_pos = [pos2id[pos_tag] for pos_tag in example.inputs_pos]
    assert len(input_pos) == len(example.inputs)

  if do_ner:
    input_ner = [ne2id[ner_tag] for ner_tag in example.inputs_ner]
    assert len(input_ner) == len(example.inputs)

  input_id, label_id = [], []
  for idx, (word_token, label) in enumerate(zip(example.inputs, example.labels)):
    label_id.append(label2id[label])
    try:
      input_id.append(word2id[word_token])
    except KeyError:
      input_id.append(word2id["<UNK>"])

  return input_id, label_id, input_pos, input_ner