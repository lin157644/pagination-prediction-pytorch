# Pagination prediction

## Introduction

In this project we we have developed a Deep Learning model designed to predict the pagination links on a web page.  
The model classifies each `<a>` and `<button>` elements in a web page into the following categories:
* PREV - previous page link
* PAGE - a link to a specific page
* NEXT - next page link
* OTHER - for elements that are not a pagination link

The model is based on the research paper: ["Large Scale Web Data API Creation via Automatic Pagination Recognition - A Case Study on Event Extraction"](https://hdl.handle.net/11296/cv453s)  
We introduce features such as URL feature from URLNet[^urlnet] and sentence embedding from Sentence Transformer[^sbert] into our model to enhance the performance beyond previous methodology.

## Datasets

We utilized the same dataset as used in ["Large Scale Web Data API Creation via Automatic Pagination Recognition - A Case Study on Event Extraction."](https://hdl.handle.net/11296/cv453s)  
This dataset is an extension of the original dataset used in the Autopager[^autopager] contains 319 pages  extracted from 109 distinct websites.

## Installation

Intall the required packages with [anaconda](https://anaconda.org/)  
```shell
conda env create -f environment.yml
```

Activate the environment
```shell
conda activate pagination-prediction-pytorch
```

## Usage

### Training

```shell
./train.sh
```

Checkpoint files will be available at `ckpt` directory after training, which can than be used by `pagination_prediction_api.py` for inference purpose.

### Test the API

```shell
./test_api.sh
```

[^urlnet]: URLNet: [**Source**](https://github.com/Antimalweb/URLNet) - [**arXiv:1802.03162**](https://arxiv.org/abs/1802.03162)
[^autopager]: Autopager: [**Source**](https://github.com/TeamHG-Memex/autopager)
[^sbert]: SentenceTransformers: [**Source**](https://github.com/UKPLab/sentence-transformers) - [**arXiv:1908.10084**](https://arxiv.org/abs/1908.10084)