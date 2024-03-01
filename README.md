# MMAPS: **End-to-End Multi-Grained Multi-Modal Attribute-Aware Product Summarization in E-commerce**

The source code and datasets for LREC-COLING 2024 paper: MMAPS: End-to-End Multi-Grained Multi-Modal Attribute-Aware Product Summarization in E-commerce [[arXiv preprint]](https://arxiv.org/abs/2308.11351).

## Folder
- `models` folder contains the image encoder `img_transformer.py`, and the overall framework `modeling_bart.py`.
- `utils` folder contains the data processing file `data_helper.py`, and the metric file `metric.py`.

## Environment

The required environment is included in `requirements.txt`.

## Data

The dataset used for experiments is a Chinese E-commerce Product summarization dataset [CEPSUM](https://github.com/hrlinlp/cepsum)

## How to run

To train the model:

```python
python main.py --mode train
```

To test the model:

```python
python main.py --mode test
```
