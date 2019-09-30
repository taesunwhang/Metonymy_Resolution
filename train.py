import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import pickle
import math
from datetime import datetime
from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, classification_report
from data_process import MetonymyProcessor
from data.data_utils import InputExamples

class MetonymyModel(object):
  def __init__(self, hparams, dataset_type):
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)
    self.dataset_type = dataset_type

  def _build_data_process(self):
    print("\t* Loading training data...")
    processors = {
      "metonymy": MetonymyProcessor,
    }

    self.processor = processors[self.hparams.task_name](self.hparams, self.dataset_type)
    self.train_examples = self.processor.get_train_examples(self.hparams.data_dir)
    self.relocar_test_examples = self.processor.get_relocar_test_examples()
    self.semeval_test_examples = self.processor.get_semeval_test_examples()

    self.word_embeddings = self.processor.get_word_embeddings()

    self.pos2id = self.processor.pos2id
    self.ne2id = self.processor.ne2id
    self.id2label = self.processor.id2label

  def _build_model(self):
    # -------------------- Model definition ------------------- #
    print('\t* Building model...')
    # Embeddings
    print('\t* Loading Word Embeddings...')

    embeddings = None
    if not self.hparams.do_use_elmo:
      embeddings = torch.tensor(self.word_embeddings, dtype=torch.float).to(self.device)

    self.model = self.hparams.model(self.hparams, self.processor.vocab, embeddings,
                                    len(self.pos2id), len(self.ne2id))

    self.model = self.model.to(self.device)

    # -------------------- Preparation for training  ------------------- #
    self.criterion = nn.CrossEntropyLoss()
    if self.dataset_type == "semeval":
      print("Weighted Cross Entropy Loss")
      class_dist_dict = self.hparams.semeval_class_dist
      print(class_dist_dict.items())
      class_weights = [sum(class_dist_dict.values()) / class_dist_dict[key] for key in class_dist_dict.keys()]
      # class_weights = [2 * max(class_weights), min(class_weights)]
      print("class_weights", class_weights)

      # sf_class_weights = [math.exp(weights) / sum([math.exp(w) for w in class_weights]) for weights in class_weights]
      # print("sf_class_weights", sf_class_weights)
      self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                mode="max",
                                                                factor=0.5,
                                                                patience=0)

  def _batch_data_to_device(self, batch_data):
    batch_inputs, batch_labels, batch_inputs_pos, batch_inputs_ner, batch_lengths, _, entities_idx, = batch_data

    batch_inputs = torch.tensor(batch_inputs).to(self.device)
    batch_labels = torch.tensor(batch_labels).to(self.device)
    batch_lengths = torch.tensor(batch_lengths).to(self.device)
    entities_idx = torch.tensor(entities_idx).to(self.device)
    batch_inputs_pos = torch.tensor(batch_inputs_pos).to(self.device)
    batch_inputs_ner = torch.tensor(batch_inputs_ner).to(self.device)

    return batch_inputs, batch_labels, batch_inputs_pos, batch_inputs_ner, batch_lengths, entities_idx

  def train(self):

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self._build_data_process()
    self._build_model()

    # total_examples
    train_data_len = int(math.ceil(len(self.train_examples)/self.hparams.train_batch_size))
    self._logger.info("Batch iteration per epoch is %d" % train_data_len)

    start_time = datetime.now().strftime('%H:%M:%S')
    self._logger.info("Start train model at %s" % start_time)
    max_eval_acc = 0
    max_eval_cls_report = None

    self.tensorboard_writer = SummaryWriter()
    for epoch_completed in range(self.hparams.num_epochs):
      self.model.train()
      loss_sum, correct_preds_sum = 0, 0

      if epoch_completed > 0:
        self.train_examples = self.processor.get_train_examples(self.hparams.data_dir)

      tqdm_batch_iterator = tqdm(range(train_data_len))
      for batch_idx in tqdm_batch_iterator:
        if not self.hparams.do_bert:
          batch_data = self.processor.get_batch_data(batch_idx, self.hparams.train_batch_size, "train")
          batch_inputs, batch_labels, batch_inputs_pos, batch_inputs_ner, batch_lengths, entities_idx = \
            self._batch_data_to_device(batch_data)
        else:
          bert_batch_data = self.processor.get_bert_batch_data(batch_idx, self.hparams.train_batch_size, "train")
          batch_inputs, batch_labels, batch_lengths, _, entities_idx = bert_batch_data
          # entity_lengths = [end - start for start, end in entities_idx]
          entities_idx = [start for start, end in entities_idx]

          batch_inputs = torch.tensor(batch_inputs).to(self.device)
          batch_labels = torch.tensor(batch_labels).to(self.device)
          batch_lengths = torch.tensor(batch_lengths).to(self.device)
          entities_idx = torch.tensor(entities_idx).to(self.device)

        mod_lengths = torch.cumsum(batch_lengths, dim=0)
        mod_entities = torch.clone(entities_idx)
        mod_entities[1:] = entities_idx[1:] + mod_lengths[0:-1]

        if self.hparams.do_bert:
          logits, labels = self.model(batch_inputs, batch_lengths, batch_labels)
        else:
          logits, labels = self.model(batch_inputs, batch_lengths, batch_labels, batch_inputs_pos, batch_inputs_ner)

        probs = logits[mod_entities, :]
        predictions = torch.argmax(probs, dim=-1)
        correct_preds = torch.sum(torch.eq(predictions.unsqueeze(-1), labels[mod_entities].unsqueeze(-1)).int())

        loss = self.criterion(probs, labels[mod_entities])
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
        self.optimizer.step()

        loss_sum += loss.item()
        correct_preds_sum += correct_preds.item()

        description = "Avg. batch proc. loss: {:.4f}" \
          .format(loss_sum / (batch_idx + 1))
        tqdm_batch_iterator.set_description(description)



      self._logger.info("-> Training loss = {:.4f}, accuracy: {:.4f}%\n"
            .format(loss_sum / train_data_len, (correct_preds_sum / len(self.train_examples))*100))

      self.tensorboard_writer.add_scalar('train/loss', (loss_sum / train_data_len), epoch_completed)
      self.tensorboard_writer.add_scalar('train/accuracy', (correct_preds_sum / len(self.train_examples)), epoch_completed)

      eval_cls_report, eval_acc = self._run_evaluate(epoch_completed)

      if eval_acc > max_eval_acc:
        max_eval_acc = eval_acc
        max_eval_cls_report = eval_cls_report

    print(max_eval_cls_report)
    self.tensorboard_writer.close()

  def _run_evaluate(self, epoch_completed):
    relocar_eval_data_len = int(math.ceil(len(self.relocar_test_examples) / self.hparams.eval_batch_size))
    semeval_eval_data_len = int(math.ceil(len(self.semeval_test_examples) / self.hparams.eval_batch_size))
    if self.dataset_type == "relocar":
      eval_dict = {"relocar_test" : (relocar_eval_data_len, self.relocar_test_examples)}
    elif self.dataset_type == "semeval":
      eval_dict = {"semeval_test" : (semeval_eval_data_len, self.semeval_test_examples)}
    elif self.dataset_type == "conll":
      eval_dict = {"relocar_test" : (relocar_eval_data_len, self.relocar_test_examples),
                   "semeval_test" : (semeval_eval_data_len, self.semeval_test_examples)}

    self.model.eval()
    with torch.no_grad():
      for key in eval_dict.keys():
        self._logger.info("Batch iteration per epoch is %d" % eval_dict[key][0])

        loss_sum, correct_preds_sum = 0, 0
        total_labels, total_preds = [], []

        for batch_idx in range(eval_dict[key][0]):
          if not self.hparams.do_bert:
            batch_data = self.processor.get_batch_data(batch_idx, self.hparams.eval_batch_size, key)
            batch_inputs, batch_labels, batch_inputs_pos, batch_inputs_ner, batch_lengths, entities_idx = \
              self._batch_data_to_device(batch_data)
          else:
            bert_batch_data = self.processor.get_bert_batch_data(batch_idx, self.hparams.eval_batch_size, key)
            batch_inputs, batch_labels, batch_lengths, _, entities_idx = bert_batch_data
            entities_idx = [start for start, end in entities_idx]

            batch_inputs = torch.tensor(batch_inputs).to(self.device)
            batch_labels = torch.tensor(batch_labels).to(self.device)
            batch_lengths = torch.tensor(batch_lengths).to(self.device)
            entities_idx = torch.tensor(entities_idx).to(self.device)

          mod_lengths = torch.cumsum(batch_lengths, dim=0)
          mod_entities = torch.clone(entities_idx)
          mod_entities[1:] = entities_idx[1:] + mod_lengths[0:-1]

          if self.hparams.do_bert:
            logits, labels = self.model(batch_inputs, batch_lengths, batch_labels)
          else:
            logits, labels = self.model(batch_inputs, batch_lengths, batch_labels, batch_inputs_pos, batch_inputs_ner)

          # probs | ground truth
          probs = logits[mod_entities, :]

          predictions = torch.argmax(probs, dim=-1)
          loss = self.criterion(probs, labels[mod_entities])

          loss_sum += loss
          correct_preds = torch.sum(torch.eq(predictions.unsqueeze(-1), labels[mod_entities].unsqueeze(-1)).int())

          # [Ground Truth] : [Model Prediction]
          # Metonymic : Literal
          # Literal : Metonymic
          total_labels.extend(labels[mod_entities].to(torch.device("cpu")))
          total_preds.extend(predictions.to(torch.device("cpu")))
          # total_entity_lengths.extend(entity_lengths)

          correct_preds_sum += correct_preds.item()

        # error_analysis(total_labels, total_preds, total_entity_lengths)
        print(classification_report(total_labels, total_preds, digits=3))
        print("{}->{} loss: {:.4f}, accuracy: {:.4f}%, f1_score : {:.4f}"
              .format(epoch_completed, key, (loss_sum / eval_dict[key][0]),
                      (correct_preds_sum / len(eval_dict[key][1]))*100, f1_score(total_labels, total_preds)))

        self.tensorboard_writer.add_scalar('evaluation/loss', (loss_sum / eval_dict[key][0]), epoch_completed)
        self.tensorboard_writer.add_scalar('evaluation/accuracy', (correct_preds_sum / len(eval_dict[key][1])), epoch_completed)

        self.scheduler.step((correct_preds_sum / len(eval_dict[key][1])))

      return classification_report(total_labels, total_preds, digits=3), (correct_preds_sum / len(eval_dict[key][1]))*100


def error_analysis(labels, preds, entity_lengths):
  assert len(labels) == len(preds) == len(entity_lengths)

  print("total eval dataset length", len(labels))

  correct_len_one_cnt = 0 # when the answer is correct : entity lengths
  correct_len_more_cnt = 0

  wrong_len_one_cnt = 0
  wrong_len_more_cnt = 0

  for label, pred, ent_len in zip(labels, preds, entity_lengths):

    if label == pred:
      # correct
      if ent_len == 1:
        correct_len_one_cnt += 1
      else:
        correct_len_more_cnt += 1

    else:
      # incorrect
      if ent_len == 1:
        wrong_len_one_cnt += 1
      else:
        wrong_len_more_cnt += 1

  print("correct : ", correct_len_one_cnt, "/", correct_len_more_cnt)
  print("incorrect : ", wrong_len_one_cnt, "/", wrong_len_more_cnt)




















