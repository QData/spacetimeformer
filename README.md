# Spacetimeformer Multivariate Forecasting

This repository contains the code for the paper, "**Long-Range Transformers for Dynamic Spatiotemporal Forecasting**", Grigsby, Wang and Qi, 2021.

![spatiotemporal_embedding](readme_media/st-graph.png)

Transformers are a high-performance approach to sequence-to-sequence timeseries forecasting. However, stacking multiple sequences into each token only allows the model to learn *temporal* relationships across time. This can ignore important *spatial* relationships between variables. Our model (nickamed "Spacetimeformer") flattens multivariate timeseries into extended sequences where each token represents the value of one variable at a given timestep. Long-Range Transformers can then learn relationships over both time and space. For much more information, please refer to our paper.

### We will be adding additional instructions, example commands and dataset links in the coming days.

## Installation 
This repository was written and tested for **python 3.7** and **pytorch 1.9.0**.

```bash
git clone https://github.com/UVA-MachineLearningBioinformatics/spacetimeformer.git
cd spacetimeformer
pip install -e .
```
This installs a python package called ``transformer_timeseries``.

## Dataset Setup
TODO

## Recreating Experiments with Our Training Script
The main training functionality for `spacetimeformer` and most baselines considered in the paper can be found in the `train.py` script. The training loop is based on the [`pytorch_lightning`](https://pytorch-lightning.rtfd.io/en/latest/) framework.

Commandline instructions for each experiment can be found using the format: ```python train.py *model* *dataset* -h```. 

Model Names:
- `linear`: a basic autoregressive linear model.
- `lstnet`: a more typical RNN/Conv1D model for multivariate forecasting. Based on the attention-free implementation of [LSTNet](https://github.com/laiguokun/LSTNet).
- `lstm`: a typical encoder-decoder LSTM without attention. We use scheduled sampling to anneal teacher forcing throughout training.
- `mtgnn`: a hybrid GNN that learns its graph structure from data. For more information refer to the [paper](https://arxiv.org/abs/2005.11650). We use the implementation from [`pytorch_geometric_temporal`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
- `timeformer`: We gave our model the arbitrary name "timeformer" early in development. The first draft of the paper ultimately called it the "spacetimeformer", but currently we have not yet changed its name in the codebase.
    - note that the "Temporal" ablation discussed in the paper is a special case of the `timeformer` model. Set the `embed_method = temporal`. Timeformer has many configurable options and we try to provide a thorough explanation with the commandline `-h` instructions.


Dataset Names:
- `metr-la` and `pems-bay`: traffic forecasting datasets. We use a very similar setup to [DCRNN](https://github.com/liyaguang/DCRNN).
- `toy2`: is the toy dataset mentioned at the beginning of our experiments section. It is heavily based on the toy dataset in [TPA-LSTM](https://arxiv.org/abs/1809.04206.).
- `asos`: Is the codebase's name for what the paper calls "NY-TX Weather."
- `solar_energy`: Is the codebase's name for what is more commonly called "AL Solar."
- `exchange`: A dataset of exchange rates. Spacetimeformer performs relatively well but this is tiny dataset of highly non-stationary data where `linear` is already a SOTA model.
- `precip`: A challenging spatial message-passing task that we have not yet been able to solve. We collected daily precipitation data from a latitude-longitude grid over the Continental United States. The multivariate sequences are sampled from a ringed "radar" configuration as shown below in green. We expand the size of the dataset by randomly moving this radar around the country.

<p align="center">
<img src="readme_media/radar_edit.png" width="220">
</p>

### Example Spacetimeformer Training Commands
Toy Dataset
```bash
python train.py timeformer toy2 --run_name spatiotemporal_toy2 \
--d_model 100 --d_ff 400 --enc_layers 4 --dec_layers 4 \
--gpus 0 1 2 3 --batch_size 32 --start_token_len 4 --n_heads 4 \
--grad_clip_norm 1 --early_stopping --trials 1
```
TODO


## Using Spacetimeformer in Other Applications
If you want to use our model in the context of other datasets or training loops, you will probably want to go a step lower than the `timeformer_model.Timeformer_Forecaster` pytorch-lightning wrapper. Please see `timeformer_model.nn.Timeformer`.
![arch-fig](readme_media/arch.png)

## Citation
If you use this model in academic work please feel free to cite our paper
```bash
@misc{grigsby2021longrange,
      title={Long-Range Transformers for Dynamic Spatiotemporal Forecasting}, 
      author={Jake Grigsby and Zhe Wang and Yanjun Qi},
      year={2021},
      eprint={2109.12218},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

![st-embed-fig](readme_media/embed.png)









