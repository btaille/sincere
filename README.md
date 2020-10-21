Let's Stop Incorrect Comparisons in End-to-End Relation Extraction!
====

Code for ["Let's Stop Incorrect Comparisons in End-to-end Relation Extraction!"](https://arxiv.org/abs/2009.10684), accepted at EMNLP 2020.


### Description
Because of differences in evaluation settings, the End-to-end Relation Extraction Literature contains incorrect 
comparisons and conclusions. 

The goal of this code is to provide a clear setup to evaluate models using either the "Strict" or 
"Boundaries" settings as defined in [(Bekoulis 2018)](https://arxiv.org/pdf/1804.07847.pdf) 
and quantify their difference.

As an example, we perform the unexplored ablation study of two recent developments: 
1. the introduction of pretrained Language Models (such as BERT or ELMo)
2. modeling the NER task as a classification for every span in the sentence instead of IOBES sequence labeling

We propose to evaluate their impact on CoNLL04 and ACE05, with no overlapping entities.

### Requirements
The code is written in Python 3.6 with the following main dependencies:

* Pytorch 1.3.1
* numpy 1.18.1
* transformers 2.4.1
* tqdm
* (optional) tensorboard 2.2.1


### Data Setup and Preprocessing

##### 1) Download Pretrained GloVe embeddings
```
glove_path='http://nlp.stanford.edu/data/glove.840B.300d.zip'
mkdir -p embeddings/glove.840B
curl -LO $glove_path
unzip glove.840B.300d.zip -d embeddings/glove.840B/
rm glove.840B.300d.zip
```

##### 2) CoNLL04
We provide CoNLL04 dataset in the `data/` folder, as formatted and used by [(Eberts 2020)](https://arxiv.org/abs/1909.07755) (<https://github.com/markus-eberts/spert>).  
It corresponds to the split released by [(Gupta 2016)](https://www.aclweb.org/anthology/C16-1239.pdf) (https://github.com/pgcool/TF-MTRNN)


##### 3) ACE05
Due to licensing issues, we do not provide the ACE05 dataset.  
The instructions and scripts to setup the dataset from (Miwa and Bansal 2016) are in the `ace_preprocessing/` folder.

### Training
Although more configurations can be tested with this code, we focused on two ablations:

1. The use of a pretrained language over non-contextualized representations with a BiLSTM : 
    - BERT Encoder : embedder is `bert-base` and no encoder flag
    - (GloVE + CharBILSTM) Embedding + BiLSTM Encoder : embedder is `word char` and encoder is `bilstm`

2. The use of a Span-level NER modules over an IOBES sequence tagging model :
    - ner_decoder is `iobes` or `span`
 
To reproduce our setup run the following commands where `$dataset` is either `conll04` or `ace05` in the `code/` folder:

##### (GloVE + charBiLSTM) + BiLSTM + IOBES NER + RE 
```bash
python train.py -ds $dataset -emb word char -enc bilstm -ner_dec iobes -d 0.1 -bs 8 -lr 5e-4 -s $seed
```

##### (GloVE + charBiLSTM) + BiLSTM + Span NER + RE 
```bash
python train.py -ds $dataset -emb word char -enc bilstm -ner_dec span -d 0.1 -bs 8 -lr 5e-4 -s $seed
```

##### BERT + IOBES NER + RE 
```bash
python train.py -ds $dataset -emb bert-base -ner_dec iobes -d 0.1 -bs 8 -lr 1e-5 -s $seed
```

##### BERT + Span NER + RE 
```bash
python train.py -ds $dataset -emb bert-base -ner_dec span -d 0.1 -bs 8 -lr 1e-5 -s $seed
```

To train on the combination of train and dev sets, add the `-m train+dev` flag after a first standard training with same parameters.


**Note:** We used seeds 0 to 4 for all our experiments. 
However, despite careful manual seeding, they are not exactly reproducible accross different GPU hardwares.

## Reference
If you find any of this work useful, please cite our paper as follows:
```
@InProceedings{taille2020sincere,
author="Taill{\'e}, Bruno and Guigue, Vincent and Scoutheeten, Geoffrey and Gallinari, Patrick",
title="Let's Stop Incorrect Comparisons in End-to-End Relation Extraction!",
booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
year="2020",
publisher="Association for Computational Linguistics",
url="https://arxiv.org/abs/2009.10684"
}
```

