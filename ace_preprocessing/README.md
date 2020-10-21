# Credits
This preprocessing code comes from [(Miwa and Bansal 2016)](https://www.aclweb.org/anthology/P16-1105.pdf)'s (<https://github.com/tticoin/LSTM-ER>)  
**/!\ We add a manual fix of one document in ACE05 dev set which is altered during tokenization**


# Requirements

* python3
* perl
* nltk (for stanford pos tagger)
* java (for stanford tools)
* zsh
* task datasets (see below)

# Links to tasks/data sets

* ACE 2004 (https://catalog.ldc.upenn.edu/LDC2005T09)
* ACE 2005 (https://catalog.ldc.upenn.edu/LDC2006T06)

Please note that ACE corpora are not free.

# Usage

## Download Stanford Core NLP & POS tagger

```
cd common
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip
unzip stanford-postagger-2015-04-20.zip
cd ..
```

## Copy and Convert each corpus 
Please set the environment variables for the directories, or directly put the directories in the following commands beforehand.

### ACE 2004

```
cp -r ${ACE2004_DIR}/*/English ace2004/
cd ace2004
zsh run.zsh
cd ..
python ann2json.py
```

### ACE 2005

```
cp -r ${ACE2005_DIR}/*/English ace2005/
cd ace2005
zsh run.zsh
cd ..
python ann2json.py
```
