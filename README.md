## SciMMIR

This is the repo for the paper [SciMMIR： Benchmarking Scientific Multi-modal Information Retrieval](https://zenodo.org/records/10521030/files/SciMMIR__Benchmarking_Scientific_Multimodal_Information_Retrieval.pdf?download=1).

<div align="center">
<img src=./imgs/Framework.png width=80% />
</div>

In this paper, we propose a novel SciMMIR benchmark and a corresponding dataset designed to address the gap in evaluating multi-modal information retrieval (MMIR) models in the scientific domain.

It is worth mentioning that we define a data hierarchical architecture of "Two subsets, Five subcategories" and use human-created keywords to classify the data (as shown in the table below).

<div align="center">
<img src=./imgs/data_architecture.png width=50% />
</div>

As shown in the table below, we conducted extensive baselines (both fine-tuning and zero-shot) within various subsets and subcategories.

![main_result](./imgs/main_result.png)

For more detailed experimental results and analysis, please refer to our paper [SciMMIR](https://zenodo.org/records/10521030/files/SciMMIR__Benchmarking_Scientific_Multimodal_Information_Retrieval.pdf?download=1).

## Dataset

Our SciMMIR benchmark dataset used in this paper contains 537K scientific image-text pairs which are extracted from the latest 6 months' papers in Arxiv (2023.05 to 2023.10), and we will continue to expand this data by extracting data from more papers in Arxiv and provide larger versions of the dataset.

The datasets can be obtained from huggingface Datasets [m-a-p/SciMMIR](https://huggingface.co/datasets/m-a-p/SciMMIR), and the following codes show how to use it:

```python
import datasets
ds_remote = datasets.load_dataset("m-a-p/SciMMIR")
test_data = ds_remote['test']
caption = test_data[0]['text']
image_type = test_data[0]['class']
image = test_data[0]['image']
```

## Fine-Tuning Model on SciMMIR Dataset

### Processing data

First, we need to get the subcategories information about our dataset by runing the following code:

```python
python classify_training_data.py
```

Then we need to processe the test and valid data:

```python
python processor_text.py --model_name  CLIP
python processor_text.py --model_name  BERT
python processor_text.py --model_name  BLIP
python processor_figs.py --model_name  CLIP
python processor_figs.py --model_name  BERT
```

All the processed data would be saved in './data/' folder.

### Training model

You can use following codes to get fine-tuned CLIP-base model:

```
 python train_main.py --training_data_len 498279
```

You can use following codes to get fine-tuned CLIP+BERT model:

```
 python train_main.py --training_data_len 498279 --Use_BERT 1 --train_batch_size 200
```

You can use following codes to  get fine-tuned BLIP-base model:

```
python train_main.py --training_data_len 498279 --model_name BLIP --train_batch_size 110 --context_length 128 --image_size 384
```

You can use following codes to  get fine-tuned BLIP-BERT model:

```
python train_main.py --training_data_len 498279 --model_name BLIP --train_batch_size 110 --context_length 128 --image_size 384 --Use_BERT 1
```

### Using  the subcategories training data

If you want to fine-tuning the model on the subcategories training data, you need to change some parameters. As for CLIP-base, you can using following codes to train model on fig_architecture data:

```
python train_main.py --image_type fig_architecture --training_data_len 13135  > CLIP_fig_architecture_log.txt 
```

As for BLIP-base,  you can using following codes to train model on fig_architecture data:

```
python train_main.py --model_name BLIP --train_batch_size 110 --context_length 128 --image_size 384 --image_type fig_architecture --training_data_len 13135
```

So, if you want to  change the training data, you just  need to change the --image_type and corresponding --training_data_len.



## Large VLMs Zero-Shot Experiments

You can use the following codes to get the text and image embedings of the LLMs that we used in paper:

```python
python LLMs_Embedding.py --model_name 'fuyu-8b'
```

and you can change the model_name to use different VLMs.

Then you can test the text and image embedding in our SciMMIR benchmark by using following codes:

```python
python test_main_LLMs.py --model_name 'fuyu-8b'
```

## Quickly Tesing Your VLMs

If you want to quickly test your fine-tuned model in our SciMMIR benchmark and don't want to change our codes, we provide a  method.

You can just use the code of Large VLMs Zero-Shot Experiments. Firstly, you need to use your modes to get the image and text  embedding. Finally, runing the test_main_LLMs.py with changing the parameter --LLM_text_embedding_saved_path as your embedding saved path.

## Potential TODOs before ACL

**TODO**: case study table

**TODO**: statistics of the paper fields (perhaps in appendix)

**TODO**: See if it's possible to further divide the "Figure Results" subsets.

## Citation

```
@misc{wu2024scimmir,
  author       = {Wu, Siwei and
                  LI, Yizhi and
                  Zhu, Kang and
                  Zhang, Ge and
                  Liang, Yiming and
                  Ma, Kaijing and
                  Xiao, Chenghao and
                  Zhang, Haoran and
                  Yang, Bohao and
                  Chen, Wenhu and
                  Huang, Wenhao and
                  Moubayed, Noura Al and
                  Fu, Jie and
                  Lin, Chenghua},
  title        = {{SciMMIR: Benchmarking Scientific Multi-modal 
                   Information Retrieval}},
  month        = jan,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10521030},
  url          = {https://doi.org/10.5281/zenodo.10521030}
}
```

