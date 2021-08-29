# UET.PJKC4.0

## KC4Align

KC4Align uses the similarity of multilingual sentence embeddings using LASER to evaluate the similarity of sentences, In conjunction with deep-translator, KC4Align works in more low-resource languages which not supported by LASER by translating it to a supported high-language

#### Installation KC4Align

You can build an environment using conda as follows:

```
* Create environment
    conda env create -f environment.yml
```
You can install [LASER Language-Agnostic Sentence Representations (LASER)](https://github.com/facebookresearch/LASER) toolkit from Facebook here which is used to embed sentences in each document.

Then set the environment variables in your workspace
```
* Set the environment variable to projects directory in your workspace
    - KC4ALIGN=${HOME}/projects/kc4align
    - LASER=${HOME}/projects/LASER
```

#### Run Tool
```
conda activate kc4align
cd $KC4ALIGN
python align.py --src_file src_path --tgt_file tgt_path -o output_dir_path
```

##### Variables Meaning

* --src_file : path to source file where each line is a paragraph

* --tgt_file : path to source file in which each line is a paragraph

* -o : output file contain pairs of sentences that alignment each other

Alignment output file is written to stdout:
```
score source_sentence taget_sentence
...
```
##### Run Sample
```
python align.py -src_file ./data/sample/vi_test.txt --tgt_file ./data/sample/lo_test.txt -o ./output
```

## Lao Sentence Tokenize

Sentence tokenization is the process of splitting text into individual sentences.

#### Dependencies

[Numpy](https://numpy.org/), tested with 1.19.2

#### Run Tool

```python
python sentence_tokenize -i input_file -o output_file
```
##### Variables Meaning
* -i : path to input lao text file 

* -o : path to the output text file