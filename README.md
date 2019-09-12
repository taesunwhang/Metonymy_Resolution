## Metonymy_Resolution
Metonymy Resolution (MR) based on Neural Approach

### Required Packages
- python 3.6.5 
- torch 1.1.0 
- allennlp 0.8.5
- pickle
- numpy

### References
- Pre-trained Language Models (e.g., BERT, XLNet, RoBERTa) are implemented based on https://github.com/huggingface/pytorch-transformers

### Dataset
1. SemEval 2007, ReLocaR
The dataset is publicly available in https://github.com/milangritta/Minimalist-Location-Metonymy-Resolution 
  - Corpus Statistics of SemEval 2007 and ReLocaR  
  <img src="/images/Metonymy_Resolution_Stats.png" width="50%">

2. Stanford NER Tagger
Stanford NER Tagger is publicly available in https://stanfordnlp.github.io/CoreNLP/index.html#download

### How to run the code

- Obtain the data files (.txt) and store them in `data/`. \
We exclude the mixed dataset since it accounts for only 2% of the datasets. \
  
  - SemEval
    - `data/semeval/semeval_literal_train.txt` 
    - `data/semeval/semeval_literal_test.txt` 
    - `data/semeval/semeval_metonymic_train.txt` 
    - `data/semeval/semeval_metonymic_test.txt`
    
  - ReLocaR
    - `data/relocar/relocar_literal_train.txt` 
    - `data/relocar/relocar_literal_test.txt` 
    - `data/relocar/relocar_metonymic_train.txt` 
    - `data/relocar/relocar_metonymic_test.txt` 
    
  - stanford-ner
    - `data/stanford-ner/stanford-ner-3.9.2.jar` 
    - `data/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz` 

- Run `python3 data/data_utils.py`
  - Pickle files are created for each dataset. 

- Run `python3 main.py --model=$model --data=$data`
  - $model = lstm, bilstm, elmo_bilstm, bert, xlnet, roberta
  - $data = semeval, relocar

### Result
1. Evaluation Results (**Accuracy**, Standard deviation)
<img src="/images/Evaluation_Results_Acc.png" width="50%">
2. Evaluation Results (**F1** Score)
<img src="/images/Evaluation_Results_F1.png" width="65%">


