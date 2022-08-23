# MREA
Codes for Chinese Sentence Semantic Matching Based on Multi-level Relevance Extraction and Aggregation for Intelligent Human-robot Interaction
The models trained by us can be downloaded from Baidu Netdisk:   
https://pan.baidu.com/s/1cmYDEJo67jRUrXtMz9uxfQ password:ls5s

## 0. Requirement
python 3.6.12  
tensorflow-gpu == 1.12.0  
keras == 2.2.4  
gensim == 3.8.3  

## 1.Data preparation
The dataset is BQ and LCQMC.

BQ: "The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification", https://www.aclweb.org/anthology/D18-1536/.

LCQMC: "LCQMC: A Large-scale Chinese Question Matching Corpus", https://aclanthology.org/C18-1166/.

## 2. Word segmentation
We utilize the tool of jieba to segment word.

## 3.Train
python BQ/train.py    
 
python LCQMC/train.py    
