# Model
This repository is the PyTorch implementation of our model in TASLP Paper 
"[Relation Extraction in Dialogues: A Deep Learning Model Based on the Generality and Specialty of Dialogue Text](https://ieeexplore.ieee.org/document/9439807?source=authoralert)".

# Requirement
```
python==3.6.10 
torch==1.5.1
tqdm==4.29.1
numpy==1.15.4
spacy==2.1.0
transformers==4.2.2
```

# Dataset

For the dataset and pretrained embeddings, please download it [here](https://github.com/nlpdata/dialogre), which are officially provided by [Dialogue-Based Relation Extraction](https://arxiv.org/abs/2004.08056)
. 
# Data Proprocessing
Run:

```
# cd code
# python3 gen_data.py 
```

# Training
In order to train the model, run:

```
# cd code
# CUDA_VISIBLE_DEVICES=0 python3 train.py --save_name dialogre --use_spemb True --use_wratt True --use_gcn True
```

# Test
After the training process, we can test the model by:

```
 CUDA_VISIBLE_DEVICES=0 python3 test.py --save_name dialogre --use_spemb True --use_wratt True --use_gcn True
```

## Related Repo

Codes are adapted from the repo of the ACL 2020 Paper 
"[Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312)".

## Citation

```
@ARTICLE{9439807,  
author={Zhou, Mengjia and Ji, Donghong and Li, Fei},  
journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
title={Relation Extraction in Dialogues: A Deep Learning Model Based on the Generality and Specialty of Dialogue Text},   
year={2021},  
volume={29},  
number={},  
pages={2015-2026},  
doi={10.1109/TASLP.2021.3082295}}
```


