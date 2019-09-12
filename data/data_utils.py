import os
import pickle
import numpy as np
import nltk
import math
from nltk import word_tokenize
from nltk.tag import StanfordNERTagger

from models.bert import tokenization_bert
from models.roberta import tokenization_roberta
from models.xlnet import tokenization_xlnet

class InputExamples(object):
		def __init__(self, inputs:list, labels:list, inputs_pos:list=None, inputs_ner:list=None,
								 seq_len:int=None, target_entity:str=None, entity_idx=None):
			self.inputs = inputs
			self.labels = labels
			self.inputs_pos = inputs_pos
			self.inputs_ner = inputs_ner
			self.seq_len = seq_len
			self.target_entity = target_entity
			self.entity_idx = entity_idx

class MetonymyBERTDataUtils(object):
	def __init__(self, dataset_type="conll", root_dir="CoNLL2003", data_type="train", tokenizer_type="bert"):
		# CoNLL2003, ReLocaR, SemEval2010
		self.literal_dir = "%s/%s_literal_%s.txt" % (root_dir, dataset_type, data_type)
		self.metonymic_dir = "%s/%s_metonymic_%s.txt" % (root_dir, dataset_type, data_type)
		self.pkl_dir = "%s/%s_%s_%s.pkl" % (root_dir, tokenizer_type, dataset_type, data_type)

		if tokenizer_type == "bert":
			self._bert_tokenizer_init()
		elif tokenizer_type == "xlnet":
			self._xlnet_tokenizer_init()
		elif tokenizer_type == "roberta":
			self._roberta_tokenizer_init()

		total_data = self._read_dataset()
		examples = self._create_input_examples(total_data)
		self._make_dataset_pkl(examples)

	def _read_dataset(self):
		total_data = []

		with open(self.literal_dir, "r", encoding="utf-8") as literal_handle:
			literal_data = [["literal", line.strip()] for line in literal_handle if len(line.rstrip()) > 0]
			total_data.extend(literal_data)

		with open(self.metonymic_dir, "r", encoding="utf-8") as metonymic_handle:
			metonymic_data = [["metonymic", line.strip()] for line in metonymic_handle if len(line.rstrip()) > 0]
			total_data.extend(metonymic_data)

		return total_data

	def _bert_tokenizer_init(self, bert_pretrained='bert-base-cased'):
		bert_pretrained_dir = "/mnt/raid5/shared/bert/pytorch/%s/" % bert_pretrained
		vocab_file_path = "%s-vocab.txt" % bert_pretrained

		self._tokenizer = tokenization_bert.BertTokenizer(
			vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path),
			do_lower_case=False
		)
		# self._bert_tokenizer = tokenization_bert.BertTokenizer.from_pretrained("bert-base-cased")
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

		self._tokenizer = tokenization_roberta.RobertaTokenizer(
			vocab_file=os.path.join(roberta_pretrained_dir, vocab_file_path),
			merges_file=os.path.join(roberta_pretrained_dir, merges_file_path)
		)
		print("roberta_tokenizer")

	def _create_input_examples(self, total_data):
		examples = []
		print("Total Examples : ", len(total_data))

		for idx, (target_label, line) in enumerate(total_data):
			print(idx, "/", len(total_data))
			target_entity, input_sequence = line.split("<SEP>")

			if target_label == "metonymic":
				target_label = "<MET>"
			else:
				target_label = "<LIT>"

			tokenized_text = []
			tokenized_labels = []

			splited_seq = input_sequence.split("<ENT>")

			tokenized_text.extend(self._tokenizer.tokenize(splited_seq[0]))
			tokenized_labels.extend(["O"] * len(tokenized_text))

			start_entity_idx = len(tokenized_text)
			assert target_entity == splited_seq[1]
			tok_target_entity = self._tokenizer.tokenize(splited_seq[1])
			tokenized_text.extend(tok_target_entity)
			tokenized_labels.extend([target_label]*len(tok_target_entity))
			end_entity_idx = len(tokenized_text)

			tokenized_text.extend(self._tokenizer.tokenize(splited_seq[2]))
			tokenized_labels.extend(["O"] * (len(tokenized_text) - len(tokenized_labels)))
			sequence_len = len(tokenized_text)

			print(tokenized_labels[start_entity_idx])

			# bert base maximum positional embeddings
			if sequence_len > 512:
				# find which direction is far from the entity
				pop_idx = 0 if start_entity_idx > (sequence_len - end_entity_idx) else -1

				# print(start_entity_idx, sequence_len - end_entity_idx + 1)
				print("original", start_entity_idx, end_entity_idx, sequence_len)
				while abs(start_entity_idx - (len(tokenized_text) - end_entity_idx)) > 1:
					print("modified", start_entity_idx, end_entity_idx, sequence_len)

					tokenized_text.pop(pop_idx)
					tokenized_labels.pop(pop_idx)
					sequence_len -= 1

					if pop_idx == 0:
						start_entity_idx -= 1
						end_entity_idx -=1

				while len(tokenized_text) > 512:
					print("modified", start_entity_idx, end_entity_idx, sequence_len)

					tokenized_text.pop(pop_idx)
					tokenized_labels.pop(pop_idx)
					sequence_len -= 1

					if pop_idx == 0:
						start_entity_idx -= 1
						end_entity_idx -=1

					pop_idx = 0 if pop_idx == -1 else -1

				print("modified", start_entity_idx, end_entity_idx, sequence_len)
			# print(tokenized_text)

			assert tokenized_labels[start_entity_idx] in ["<MET>", "<LIT>"]
			assert tokenized_labels[end_entity_idx - 1] in ["<MET>", "<LIT>"]

			assert len(tokenized_text) == len(tokenized_labels) == sequence_len

			examples.append(
				InputExamples(inputs=tokenized_text,
											labels=tokenized_labels,
											seq_len=sequence_len,
											target_entity=target_entity,
											entity_idx=(start_entity_idx, end_entity_idx))
			)

			if idx == 0:
				print(tokenized_text)
				print(tokenized_labels)
				print(target_entity)
				print((start_entity_idx, end_entity_idx))

		return examples

	def _make_dataset_pkl(self, examples):
		with open(self.pkl_dir, "wb") as pkl_handle:
			pickle.dump(examples, pkl_handle)

class MetonymyDataUtils(object):
	def __init__(self, dataset_type="conll", root_dir="CoNLL2003", data_type="train"):
		"""
		:param dataset_type: conll, semeval, relocar
		"""

		self.stanford_ner_tagger = StanfordNERTagger(
			'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
			'stanford-ner/stanford-ner-3.9.2.jar',
			encoding='utf-8'
		)

		#CoNLL2003, ReLocaR, SemEval2010
		self.literal_dir = "%s/%s_literal_%s.txt" % (root_dir, dataset_type, data_type)
		self.metonymic_dir = "%s/%s_metonymic_%s.txt" % (root_dir, dataset_type, data_type)
		self.pkl_dir = "%s/%s_%s.pkl" % (root_dir, dataset_type, data_type)
		self.vocab = None

		if data_type == "train":
			self.vocab_dir = "%s/%s_vocab.txt" % (root_dir, dataset_type)
			self.vocab = set()

		total_data = self._read_dataset()
		examples = self._create_input_examples(total_data)
		self._make_dataset_pkl(examples)

		if data_type == "train": self._make_vocab_file()

	def _read_dataset(self):
		total_data = []

		with open(self.literal_dir, "r", encoding="utf-8") as literal_handle:
			literal_data = [["literal", line.strip()] for line in literal_handle if len(line.rstrip()) > 0]
			total_data.extend(literal_data)

		with open(self.metonymic_dir, "r", encoding="utf-8") as metonymic_handle:
			metonymic_data = [["metonymic", line.strip()] for line in metonymic_handle if len(line.rstrip()) > 0]
			total_data.extend(metonymic_data)

		return total_data

	def _create_input_examples(self, total_data):
		examples = []
		print("Total Examples : ", len(total_data))

		for idx, (target_label, line) in enumerate(total_data):
			print(idx, "/", len(total_data))
			target_entity, input_sequence = line.split("<SEP>")

			if target_label == "metonymic": target_label = "<MET>"
			else: target_label = "<LIT>"

			tokenized_text = []
			tokenized_labels = []
			tokenized_pos = []
			tokenize_ner = []

			splited_seq = input_sequence.split("<ENT>")

			tokenized_text.extend(word_tokenize(splited_seq[0]))
			tokenized_labels.extend(["O"]*len(tokenized_text))

			entity_idx = len(tokenized_text)
			assert target_entity == splited_seq[1]
			tokenized_text.extend([splited_seq[1]])
			tokenized_labels.append(target_label)

			tokenized_text.extend(word_tokenize(splited_seq[2]))
			tokenized_labels.extend(["O"] * (len(tokenized_text) - len(tokenized_labels)))
			sequence_len = len(tokenized_text)

			pos_tags = nltk.pos_tag(tokenized_text)
			for (pos_word, pos_tag) in pos_tags:
				tokenized_pos.append(pos_tag)

			ner_tags = self.stanford_ner_tagger.tag(tokenized_text)
			for (ner_word, ner_tag) in ner_tags:
				tokenize_ner.append(ner_tag)

			assert len(tokenized_text) == len(tokenized_labels)
			assert len(tokenized_text) == len(tokenize_ner)
			assert len(tokenized_text) == len(tokenized_pos)

			# add vocabulary
			if self.vocab is not None:
				for token in tokenized_text:
					self.vocab.add(token)

			examples.append(
				InputExamples(inputs=tokenized_text,
											labels=tokenized_labels,
											inputs_pos=tokenized_pos,
											inputs_ner=tokenize_ner,
											seq_len=sequence_len,
											target_entity=target_entity,
											entity_idx=entity_idx)
			)

		return examples

	def _make_dataset_pkl(self, examples):
		with open(self.pkl_dir, "wb") as pkl_handle:
			pickle.dump(examples, pkl_handle)

	def _make_vocab_file(self):
		with open(self.vocab_dir, "w", encoding="utf8") as vocab_handle:
			vocab = list(self.vocab)
			vocab.insert(0, "<PAD>")
			vocab.append("<UNK>")
			for word_token in vocab:
				vocab_handle.write(word_token + "\n")

class GLoVEProcessor(object):
	def __init__(self, vocab_path, trimmed_path):
		glove_path = "/home/taesun/Documents/glove.840B.300d.txt"

		# self.write_vocab_file(self.vocab, self.vocab_path)
		total_vocab, word2id = self.load_vocab(vocab_path)
		glove_vocab = self.load_glove_vocab(glove_path)

		vocab_intersection = set(total_vocab) & glove_vocab
		print("%s vocab is in Glove" % len(vocab_intersection))

		self.export_trimmed_glove_vectors(word2id, glove_path, trimmed_path, 300)

	def load_vocab(self, vocab_path):
		with open(vocab_path, "r", encoding="utf8") as vocab_handle:
			vocab = [word.strip() for word in vocab_handle if len(word.strip()) > 0]

		word2id = dict()
		for idx, word in enumerate(vocab):
			word2id[word] = idx

		return vocab, word2id

	def load_glove_vocab(self, glove_path):
		"""
		Args:
				filename: path to the glove vectors hparams.glove_path
		"""
		print("Building glove vocab...")
		glove_vocab = set()
		with open(glove_path, "r", encoding='utf-8') as f_handle:
			for line in f_handle:
				word = line.strip().split(' ')[0]
				glove_vocab.add(word)

		print("Getting Glove Vocabulary is done. %d tokens" % len(glove_vocab))

		return glove_vocab

	def export_trimmed_glove_vectors(self, word2id, glove_path, trimmed_path, dim):
		"""
		Saves glove vectors in numpy array
		Args:
				vocab: dictionary vocab[word] = index
				glove_filename: a path to a glove file
				trimmed_filename: a path where to store a matrix in npy
				dim: (int) dimension of embeddings
		"""
		embeddings = np.random.uniform(low=-1, high=1, size=(len(word2id), dim))
		print(embeddings.shape)

		with open(glove_path, encoding='utf-8') as f:
			for line in f:
				line = line.strip().split(' ')
				word = line[0]
				embedding = [float(x) for x in line[1:]]

				if len(embedding) < 2:
					continue

				if word in word2id:
					word_idx = word2id[word]
					embeddings[word_idx] = np.asarray(embedding)

		np.savez_compressed(trimmed_path, embeddings=embeddings)

if __name__ == '__main__':
	# glove, elmo
	MetonymyDataUtils(dataset_type="conll",root_dir="conll", data_type="train")
	MetonymyDataUtils(dataset_type="relocar",root_dir="relocar", data_type="train")
	MetonymyDataUtils(dataset_type="relocar",root_dir="relocar", data_type="test")
	MetonymyDataUtils(dataset_type="semeval",root_dir="semeval", data_type="train")
	MetonymyDataUtils(dataset_type="semeval",root_dir="semeval", data_type="test")

	# glove
	GLoVEProcessor("conll/conll_vocab.txt", "conll/conll_glove.840B.300d.npz")
	GLoVEProcessor("relocar/relocar_vocab.txt", "relocar/relocar_glove.840B.300d.npz")
	GLoVEProcessor("semeval/semeval_vocab.txt", "semeval/semeval_glove.840B.300d.npz")

	# bert
	MetonymyBERTDataUtils(dataset_type="conll",root_dir="conll", data_type="train", tokenizer_type="bert")
	MetonymyBERTDataUtils(dataset_type="relocar",root_dir="relocar", data_type="train", tokenizer_type="bert")
	MetonymyBERTDataUtils(dataset_type="relocar",root_dir="relocar", data_type="test", tokenizer_type="bert")
	MetonymyBERTDataUtils(dataset_type="semeval",root_dir="semeval", data_type="train", tokenizer_type="bert")
	MetonymyBERTDataUtils(dataset_type="semeval",root_dir="semeval", data_type="test", tokenizer_type="bert")

	# xlnet
	MetonymyBERTDataUtils(dataset_type="conll",root_dir="conll", data_type="train", tokenizer_type="xlnet")
	MetonymyBERTDataUtils(dataset_type="relocar",root_dir="relocar", data_type="train", tokenizer_type="xlnet")
	MetonymyBERTDataUtils(dataset_type="relocar",root_dir="relocar", data_type="test", tokenizer_type="xlnet")
	MetonymyBERTDataUtils(dataset_type="semeval",root_dir="semeval", data_type="train", tokenizer_type="xlnet")
	MetonymyBERTDataUtils(dataset_type="semeval",root_dir="semeval", data_type="test", tokenizer_type="xlnet")

	# roberta
	MetonymyBERTDataUtils(dataset_type="conll",root_dir="conll", data_type="train", tokenizer_type="roberta")
	MetonymyBERTDataUtils(dataset_type="relocar",root_dir="relocar", data_type="train", tokenizer_type="roberta")
	MetonymyBERTDataUtils(dataset_type="relocar",root_dir="relocar", data_type="test", tokenizer_type="roberta")
	MetonymyBERTDataUtils(dataset_type="semeval",root_dir="semeval", data_type="train", tokenizer_type="roberta")
	MetonymyBERTDataUtils(dataset_type="semeval",root_dir="semeval", data_type="test", tokenizer_type="roberta")