from models.base_classes import *
from models.bert import modeling_bert
from models.roberta import modeling_roberta
from models.xlnet import modeling_xlnet

import torch.nn as nn

class BiLSTM(nn.Module):
  def __init__(self, hparams, vocab=None, word_embeddings=None, pos_vocab_size=None, ner_class_num=None):
    super(BiLSTM, self).__init__()
    self.hparams = hparams

    if self.hparams.do_use_elmo:
      self._elmo = ELMoEmbeddings(self.hparams.elmo_options_file, self.hparams.elmo_weight_file, vocab)
      word_embedding_dim = self.hparams.elmo_embedding_dim

    else:
      self._word_embedding = nn.Embedding(len(vocab),
                                          self.hparams.embedding_dim,
                                          padding_idx=0,
                                          _weight=word_embeddings)
      word_embedding_dim = self.hparams.embedding_dim

    pos_embedding_dim = 0
    if self.hparams.do_pos:
      print("BiLSTM with POS Features")
      self._pos_embedding = nn.Embedding(pos_vocab_size,
                                         self.hparams.pos_embedding_dim,
                                         padding_idx=0)
      pos_embedding_dim = self.hparams.pos_embedding_dim

    ner_embedding_dim = 0
    if self.hparams.do_ner:
      print("BiLSTM with NER Feature")
      self._ner_embedding = nn.Embedding(ner_class_num,
                                         self.hparams.ner_embedding_dim,
                                         padding_idx=0)
      ner_embedding_dim = self.hparams.ner_embedding_dim

    self._rnn_encoder = RNNEncoder(
      rnn_type=nn.LSTM,
      input_size=word_embedding_dim + pos_embedding_dim + ner_embedding_dim,
      hidden_size=self.hparams.rnn_hidden_dim,
      num_layers=self.hparams.rnn_depth,
      bias=True,
      dropout=1-self.hparams.dropout_keep_prob,
      bidirectional=True
    )

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.rnn_hidden_dim * 2, self.hparams.rnn_hidden_dim),
      nn.Tanh(),
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.rnn_hidden_dim, len(self.hparams.output_classes))
    )

  def forward(self, batch_sequence, sequence_lengths, batch_label, batch_inputs_pos, batch_inputs_ner):
    if self.hparams.do_use_elmo:
      final_embedded = self._elmo(batch_sequence)
    else:
      final_embedded = self._word_embedding(batch_sequence)

    if self.hparams.do_pos:
      embedded_batch_pos = self._pos_embedding(batch_inputs_pos)
      final_embedded = torch.cat([final_embedded, embedded_batch_pos], dim=-1)

    if self.hparams.do_ner:
      embedded_batch_ner = self._ner_embedding(batch_inputs_ner)
      final_embedded = torch.cat([final_embedded, embedded_batch_ner], dim=-1)

    rnn_outputs = self._rnn_encoder(final_embedded, sequence_lengths)
    rnn_reshaped_outputs = rnn_outputs.view(-1, self.hparams.rnn_hidden_dim*2)

    batch_seq_mask = get_mask(batch_label, sequence_lengths)
    batch_seq_mask = batch_seq_mask.view(-1)

    logits = rnn_reshaped_outputs[batch_seq_mask.nonzero(),:]
    logits = logits.squeeze(dim=1)
    logits = self._classification(logits)

    labels = batch_label.view(-1)[batch_seq_mask.nonzero()].squeeze(1)
    labels = labels - 1

    return logits, labels

class UniLSTM(nn.Module):
  def __init__(self, hparams, vocab=None, word_embeddings=None, pos_vocab_size=None, ner_class_num=None):

    super(UniLSTM, self).__init__()
    self.hparams = hparams

    self._word_embedding = nn.Embedding(len(vocab),
                                        self.hparams.embedding_dim,
                                        padding_idx=0,
                                        _weight=word_embeddings)

    pos_embedding_dim = 0
    if self.hparams.do_pos:
      print("LSTM with POS Feature")
      self._pos_embedding = nn.Embedding(pos_vocab_size,
                                         self.hparams.pos_embedding_dim,
                                         padding_idx=0)
      pos_embedding_dim = self.hparams.pos_embedding_dim

    ner_embedding_dim = 0
    if self.hparams.do_ner:
      print("LSTM with NER Feature")
      self._ner_embedding = nn.Embedding(ner_class_num,
                                         self.hparams.ner_embedding_dim,
                                         padding_idx=0)
      ner_embedding_dim = self.hparams.ner_embedding_dim

    self._rnn_encoder = RNNEncoder(
      rnn_type=nn.LSTM,
      input_size=self.hparams.embedding_dim + pos_embedding_dim + ner_embedding_dim,
      hidden_size=self.hparams.rnn_hidden_dim,
      num_layers=self.hparams.rnn_depth,
      bias=True,
      dropout=1-self.hparams.dropout_keep_prob,
      bidirectional=False
    )

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.rnn_hidden_dim, int(self.hparams.rnn_hidden_dim / 2)),
      nn.Tanh(),
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(int(self.hparams.rnn_hidden_dim / 2), len(self.hparams.output_classes))
    )

  def forward(self, batch_sequence, sequence_lengths, batch_label, batch_inputs_pos, batch_inputs_ner):

    final_embedded = self._word_embedding(batch_sequence)

    if self.hparams.do_pos:
      embedded_batch_pos = self._pos_embedding(batch_inputs_pos)
      final_embedded = torch.cat([final_embedded, embedded_batch_pos], dim=-1)

    if self.hparams.do_ner:
      embedded_batch_ner = self._ner_embedding(batch_inputs_ner)
      final_embedded = torch.cat([final_embedded, embedded_batch_ner], dim=-1)

    rnn_outputs = self._rnn_encoder(final_embedded, sequence_lengths)
    rnn_reshaped_outputs = rnn_outputs.view(-1, self.hparams.rnn_hidden_dim)

    batch_seq_mask = get_mask(batch_sequence, sequence_lengths)
    batch_seq_mask = batch_seq_mask.view(-1)

    logits = rnn_reshaped_outputs[batch_seq_mask.nonzero(), :]
    logits = logits.squeeze(dim=1)
    logits = self._classification(logits)

    labels = batch_label.view(-1)[batch_seq_mask.nonzero()].squeeze(1)
    labels = labels - 1

    return logits, labels


class BERTEncoder(nn.Module):
  def __init__(self, hparams, vocab=None, word_embeddings=None, pos_vocab_size=None, ner_class_num=None):
    super(BERTEncoder, self).__init__()
    self.hparams = hparams
    self._bert_model = modeling_bert.BertModel.from_pretrained(self.hparams.bert_pretrained)

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.bert_hidden_dim, int(self.hparams.bert_hidden_dim / 2)),
      nn.Tanh(),
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(int(self.hparams.bert_hidden_dim / 2), len(self.hparams.output_classes))
    )
    # self.init_weights()

  def forward(self, batch_sequence, sequence_lengths, batch_label):
    segment_tensors = torch.zeros(batch_sequence.size(), dtype=torch.long).cuda()

    bert_outputs, _ = self._bert_model(batch_sequence, segment_tensors)
    bert_reshaped_outputs = bert_outputs.view(-1, self.hparams.bert_hidden_dim)

    batch_seq_mask = get_mask(batch_sequence, sequence_lengths)
    batch_seq_mask = batch_seq_mask.view(-1)

    logits = bert_reshaped_outputs[batch_seq_mask.nonzero(), :]
    logits = logits.squeeze(dim=1)
    logits = self._classification(logits)

    labels = batch_label.view(-1)[batch_seq_mask.nonzero()].squeeze(1)
    labels = labels - 1

    return logits, labels

class XLNetEncoder(nn.Module):
  def __init__(self, hparams, vocab=None, word_embeddings=None, pos_vocab_size=None, ner_class_num=None):
    super(XLNetEncoder, self).__init__()
    self.hparams = hparams
    self._xlnet_model = modeling_xlnet.XLNetModel.from_pretrained(self.hparams.xlnet_pretrained)

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.xlent_hidden_dim, int(self.hparams.xlent_hidden_dim / 2)),
      nn.Tanh(),
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(int(self.hparams.xlent_hidden_dim / 2), len(self.hparams.output_classes))
    )

  def forward(self, batch_sequence, sequence_lengths, batch_label):
    segment_tensors = torch.zeros(batch_sequence.size(), dtype=torch.long).cuda()

    xlnet_outputs, _ = self._xlnet_model(batch_sequence, segment_tensors)
    xlnet_reshaped_outputs = xlnet_outputs.view(-1, self.hparams.xlent_hidden_dim)

    batch_seq_mask = get_mask(batch_sequence, sequence_lengths, self.hparams.pad_idx)
    batch_seq_mask = batch_seq_mask.view(-1)

    logits = xlnet_reshaped_outputs[batch_seq_mask.nonzero(), :]
    logits = logits.squeeze(dim=1)
    logits = self._classification(logits)

    labels = batch_label.view(-1)[batch_seq_mask.nonzero()].squeeze(1)
    labels = labels - 1

    return logits, labels

class RobertaEncoder(nn.Module):
  def __init__(self, hparams, vocab=None, word_embeddings=None, pos_vocab_size=None, ner_class_num=None):
    super(RobertaEncoder, self).__init__()
    self._pad_idx = 1
    self.hparams = hparams
    self._roberta_model = modeling_roberta.RobertaModel.from_pretrained(self.hparams.roberta_pretrained)

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.bert_hidden_dim, int(self.hparams.bert_hidden_dim / 2)),
      nn.Tanh(),
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(int(self.hparams.bert_hidden_dim / 2), len(self.hparams.output_classes))
    )

  def forward(self, batch_sequence, sequence_lengths, batch_label):
    segment_tensors = torch.zeros(batch_sequence.size(), dtype=torch.long).cuda()

    bert_outputs, _ = self._roberta_model(batch_sequence, segment_tensors)
    bert_reshaped_outputs = bert_outputs.view(-1, self.hparams.bert_hidden_dim)

    batch_seq_mask = get_mask(batch_sequence, sequence_lengths, self.hparams.pad_idx)
    batch_seq_mask = batch_seq_mask.view(-1)

    logits = bert_reshaped_outputs[batch_seq_mask.nonzero(), :]
    logits = logits.squeeze(dim=1)
    logits = self._classification(logits)

    labels = batch_label.view(-1)[batch_seq_mask.nonzero()].squeeze(1)
    labels = labels - 1

    return logits, labels