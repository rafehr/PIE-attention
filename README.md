# PIE-attention

This repo contains the implementation of the work described in [An Analysis of Attention in German Verbal Idiom Disambiguation](http://www.lrec-conf.org/proceedings/lrec2022/workshops/MWE/pdf/2022.mwe2022-1.5.pdf).

To use it, make sure to install the packages listed in [requirements.txt](https://github.com/rafehr/PIE-attention/blob/main/requirements.txt). Furthermore, in order to parse the sentences, you will need to download the German transformer pipline for spacy with:

```python -m spacy download de_dep_news_trf```

After that, you can perform the following steps:

1. Run```python build.py --corpus_dir <path_to_COLF-VID_1.0_data>``` to perform a balanced 70/15/15 split of the data set. This will create the folder ```data``` with three subfolders ```train```, ```dev``` and ```test``` with the following files in each folder: 

- sentences.txt
- pie_idxs.txt (The indices of the PIE components)
- labels.txt
- pos_tags.txt
- sent_ids.txt
- pie_types.txt (The PIE types an instance belongts to)

Additionaly it will create a file containing the vocabulary (```vocab.txt```) and the label set (```label_set.txt```) as well as a file with the numbers of instances per PIE type (```num_instances_per_type.json```).

2. The model takes [FastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) as input and expects a file with embeddings for all words in the vocab. To create this file, download the binary file for the [German model](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz) and then run ```python fetch_fasttext_embs.py --model <path_to_fasttext_binary_file> --vocab_path <path_to_vocab_file>```.

3. Run ```parse_sents``` with ```python parse_sents``` to create to additional files:

- heads.txt (The head for evey word in a sentence)
- deprels.txt (The dependency relations)

4. To train a model run ```python train.py --model_name <model_name>```. It will get saved in the directory ```trained_models```.
5. To evaluate a model run ```python evaluation.py --model_name <model_name>```. This will not only evaluate the model, but also collect the attention properties of the individual data points which are saved in the directory ```stats``` as ```attn_stats.json```. These can then be used as a basis for statistical analysis. 
