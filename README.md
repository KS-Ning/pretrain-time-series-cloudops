# Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain

Official code repository for the paper "Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain". 
Check out our [paper](https://arxiv.org/abs/2310.05063) for more details. Accompanying datasets can be found [here](https://huggingface.co/datasets/Salesforce/cloudops_tsf).

# Usage

Install the required packages.

Torch experiments:
```bash
pip install -r requirements/requirements-pytorch.txt
```

statsforecast experiments:
```bash
pip install -r requirements/requirements-stats.txt
```

## Dataset

Easily load and access the dataset from Hugging Face Hub:

```bash
from datasets import load_dataset

ds = load_dataset(
    "Salesforce/cloudops_tsf",
    "azure_vm_traces_2017",  # "borg_cluster_data_2011", "alibaba_cluster_trace_2018"
    split=None,  # "train_test", "pretrain"
)
```

## Benchmark Experiments

We use [Hydra](https://hydra.cc/) for config management.

### Deep Learning Models

Run the hyperparameter tuning script:
```bash
python -m benchmark.benchmark_exp model_name=MODEL_NAME dataset_name=DATASET
```
* where `MODEL_SIZE` is one of: `TemporalFusionTransformer`, `Autoformer`, `FEDformer`, `NSTransformer`, `PatchTST`, `LinearFamily`, `DeepTime`, `TimeGrad`, or `DeepVAR`.
* `DATASET` is one of `azure_vm_traces_2017`, `borg_cluster_data_2011`, or `alibaba_cluster_trace_2018`.

After hyperparameter tuning, run the test script:
```bash
python -m benchmark.benchmark_exp model_name=MODEL_NAME dataset_name=DATASET test=true
```
* where `MODEL_SIZE` is one of: `TemporalFusionTransformer`, `Autoformer`, `FEDformer`, `NSTransformer`, `PatchTST`, `LinearFamily`, `DeepTime`, `TimeGrad`, or `DeepVAR`.
* `DATASET` is one of `azure_vm_traces_2017`, `borg_cluster_data_2011`, or `alibaba_cluster_trace_2018`.
* training logs and checkpoints will be saved in `outputs/benchmark_exp`

### Statistical Models
```bash
python -m benchmark.stats_exp DATASET --models MODEL_1 MODEL_2
```
* `DATASET` is one of `azure_vm_traces_2017`, `borg_cluster_data_2011`, or `alibaba_cluster_trace_2018`.
* `MODEL_1`, `MODEL_2` is a list of models you want to run, from `naive`, `auto_arima`, `auto_ets`, `auto_theta`, `multivariate_naive`, or `var`.

## Pre-training Experiments
Run the pre-training script:
```bash
python -m pretraining.pretrain_exp backbone=BACKBONE size=SIZE ++data.dataset_name=DATASET
```
* where the options for `BACKBONE`, `SIZE` options can be found in `conf/backbone` and `conf/size` respectively.
* `DATASET` is one of `azure_vm_traces_2017`, `borg_cluster_data_2011`, or `alibaba_cluster_trace_2018`.
* see `confg/pretrain.yaml` for more details on the options.
* training logs and checkpoints will be saved in `outputs/pretrain_exp`

Run the forecast script:
```bash
python -m pretraining.forecast_exp backbone=BACKBONE forecast=FORECAST size=SIZE ++data.dataset_name=DATASET
```
* where the options for ```BACKBONE```, ```FORECAST```, ```SIZE``` options can be found in ```conf/backbone```, ```conf/forecast```, and ```conf/size``` respectively.
* `DATASET` is one of `azure_vm_traces_2017`, `borg_cluster_data_2011`, or `alibaba_cluster_trace_2018`.
* see ```confg/forecast.yaml``` for more details on the options.
* training logs and checkpoints will be saved in `outputs/forecast_exp`

# Citation
If you find the paper or the source code useful to your projects, please cite the following bibtex:
<pre>
@article{woo2023pushing,
  title={Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain},
  author={Woo, Gerald and Liu, Chenghao and Kumar, Akshat and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2310.05063},
  year={2023}
}
</pre>
