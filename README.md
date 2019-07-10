# Latent ODEs for Irregularly-Sampled Time Series

Code for the paper:
> Yulia Rubanova, Ricky Chen, David Duvenaud. "Latent ODEs for Irregularly-Sampled Time Series" (2019)
[[arxiv]](https://arxiv.org/abs/1907.03907)

<p align="center">
<img align="middle" src="./assets/viz.gif" width="800" />
</p>

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Experiments on different datasets

By default, the dataset are downloadeded and processed when script is run for the first time. 

Raw datasets: 
[[MuJoCo]](http://www.cs.toronto.edu/~rtqichen/datasets/HopperPhysics/training.pt)
[[Physionet]](https://physionet.org/physiobank/database/challenge/2012/)
[[Human Activity]](https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity/)

To generate MuJoCo trajectories from scratch, [DeepMind Control Suite](https://github.com/deepmind/dm_control/) is required


* Toy dataset of 1d periodic functions
```
python3 run_models.py --niters 500 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 
```

* MuJoCo

```
python3 run_models.py --niters 300 -n 10000 -l 15 --dataset hopper --latent-ode --rec-dims 30 --gru-units 100 --units 300 --gen-layers 3 --rec-layers 3
```

* Physionet (discretization by 1 min)
```
python3 run_models.py --niters 100 -n 8000 -l 20 --dataset physionet --latent-ode --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.016 --classif

```

* Human Activity
```
python3 run_models.py --niters 200 -n 10000 -l 15 --dataset activity --latent-ode --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 500 --gru-units 50 --classif  --linear-classif

```


### Running different models

* ODE-RNN
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --ode-rnn
```

* Latent ODE with ODE-RNN encoder
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode
```

* Latent ODE with ODE-RNN encoder and poisson likelihood
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode --poisson
```

* Latent ODE with RNN encoder (Chen et al, 2018)
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode --z0-encoder rnn
```

* RNN-VAE
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --rnn-vae
```

*  Classic RNN
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --classic-rnn
```

* GRU-D

GRU-D consists of two parts: input imputation (--input-decay) and exponential decay of the hidden state (--rnn-cell expdecay)

```
python3 run_models.py --niters 500 -n 100  -b 30 -l 10 --dataset periodic  --classic-rnn --input-decay --rnn-cell expdecay
```


### Making the visualization
```
python3 run_models.py --niters 100 -n 5000 -b 100 -l 3 --dataset periodic --latent-ode --noise-weight 0.5 --lr 0.01 --viz --rec-layers 2 --gen-layers 2 -u 100 -c 30
```
