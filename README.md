

# Semantic Attention Network for Intuitive Information Retrieval 
Pytorch implementation integrating the Tree-LSTM network from "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks" (http://arxiv.org/abs/1503.00075, Kai Sheng Tai et al.) and "A Structured Self-Attentive
Sentence Embedding" (https://arxiv.org/pdf/1703.03130.pdf, Zhouhan Lin et al.).

The model achieves about the same test accuracy (86.85%) on the SICK dataset as the Tree-LSTM alone (86.76%), and additionally provides methods for understanding how the network is learning semantics, as well as information compression via learned attention weights. State of the art results achieve 88.5% using transfer learning (https://arxiv.org/pdf/1705.02364.pdf).

A write up of this work can be found at https://journals.mcmaster.ca/mjep/article/view/1627/1230.

### Requirements
- Python 2.7 (tested on **2.7.12**)
- [PyTorch](http://pytorch.org/) (tested on **0.1.12**)
- [tqdm](https://github.com/tqdm/tqdm)
- Java >= 8 (for Stanford CoreNLP utilities)

### Usage
 - First run the script `./fetch_and_preprocess.sh`, which, as the name suggests, does two things:
     - Fetch data, such as:
         - [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) (semantic relatedness task)
         - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!
         - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) and [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)
     - Preprocess data, i.e. generate dependency parses using [Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).


### Acknowledgements
Thanks to Riddhiman Dasgupta for his open source Pytorch implementation of the dependency tree-LSTM and Haoyue Shi for his open source Pytorch implementation of the self-structured attention mechansism.

https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding
https://github.com/dasguptar/treelstm.pytorch

### License
MIT
