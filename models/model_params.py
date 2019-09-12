from collections import defaultdict
from models.metonymy_baselines import *

BASE_PARAMS = defaultdict(
  # lambda: None,  # Set default value to None.
  # GPU params
  gpu_num = [0],

  # Input params
  train_batch_size=16,
  eval_batch_size=16,

  # Training params
  learning_rate=3e-5,
  training_shuffle_num=50,
  dropout_keep_prob=0.8,
  num_epochs=20,
  embedding_dim=300,
  elmo_embedding_dim = 1024,
  pos_embedding_dim=100,
  ner_embedding_dim=100,
  rnn_hidden_dim=256,
  rnn_depth=2,
  output_classes = ["<MET>", "<LIT>"],
  max_gradient_norm=10.0,

  max_position_embeddings=512,
  num_hidden_layers=12,
  num_attention_heads=12,
  intermediate_size=3072,
  attention_probs_dropout_prob=0.1,
  layer_norm_eps=1e-12,

  # Input Config
  semeval_class_dist = {"<MET>": 173, "<LIT>" : 737},

  # Train Model Config
  task_name="metonymy",
  do_pos=True,
  do_ner=True,
  do_use_elmo=False,
  do_bert=False,
  do_roberta=False,

  # Need to change to train...(e.g.data dir, config dir, vocab dir, etc.)
  glove_vocab_dir="data/%s/%s_vocab.txt",
  glove_embedding_path = "data/%s/%s_glove.840B.300d.npz",
  root_dir="./runs/",
  data_dir="data/%s/",
  pad_idx=0,

  train_pkl="%s_train.pkl",
  relocar_test_pkl="relocar_test.pkl",
  semeval_test_pkl="semeval_test.pkl"
)

BiLSTM_PARAMS = BASE_PARAMS.copy()
BiLSTM_PARAMS.update(
  model=BiLSTM,
)

LSTM_PARAMS = BASE_PARAMS.copy()
LSTM_PARAMS.update(
  model=UniLSTM
)

BILSTM_ELMO_PARAMS = BASE_PARAMS.copy()
BILSTM_ELMO_PARAMS.update(
  do_use_elmo=True,
  elmo_options_file = "/mnt/raid5/shared/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
  elmo_weight_file = "/mnt/raid5/shared/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
  model=BiLSTM,
  eval_batch_size=16
)

BERT_PARAMS = BASE_PARAMS.copy()
BERT_PARAMS.update(
  pad_idx=0,
  do_bert=True,
  do_roberta=False,
  do_xlnet=False,
  bert_hidden_dim=768,
  num_epochs=20,
  train_pkl="bert_%s_train.pkl",
  relocar_test_pkl="bert_relocar_test.pkl",
  semeval_test_pkl="bert_semeval_test.pkl",
  bert_pretrained="bert-base-cased",
  model=BERTEncoder,
  learning_rate=2e-5,
)

RoBERTa_PARAMS = BASE_PARAMS.copy()
RoBERTa_PARAMS.update(
  pad_idx=1,
  do_bert=True,
  do_xlnet=False,
  do_roberta=True,
  bert_hidden_dim=768,
  num_epochs=20,
  train_pkl="roberta_%s_train.pkl",
  relocar_test_pkl="roberta_relocar_test.pkl",
  semeval_test_pkl="roberta_semeval_test.pkl",
  roberta_pretrained="roberta-base",
  model=RobertaEncoder,
  learning_rate=2e-5,
)

XLNET_PARAMS = BASE_PARAMS.copy()
XLNET_PARAMS.update(
  pad_idx=5,
  do_bert=True,
  do_xlnet=True,
  do_roberta=False,
  eval_batch_size=16,
  train_batch_size=16,
  xlent_hidden_dim=768,
  num_epochs=20,
  train_pkl="xlnet_%s_train.pkl",
  relocar_test_pkl="xlnet_relocar_test.pkl",
  semeval_test_pkl="xlnet_semeval_test.pkl",
  xlnet_pretrained="xlnet-base-cased",
  model=XLNetEncoder,
  learning_rate=2e-5,
)