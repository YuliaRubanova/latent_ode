# Latent ODEs for Irregularly-Sampled Time Series

Code for the paper
> Yulia Rubanova, Ricky Chen, David Duvenaud. "Latent ODEs for Irregularly-Sampled Time Series" (2019)


## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Experiments

### Datasets
Toy dataset of 1d periodic functions: 
```
python3 run_models.py --niters 500 -n 1000   -b 50  -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 --lr 0.01
```

MuJoCo:
By default, the dataset is downloaded from [here](http://www.cs.toronto.edu/~rtqichen/datasets/HopperPhysics/training.pt)
[DeepMind Control Suite](https://github.com/deepmind/dm_control/) is required to generate trajectories from scratch
```
python3 run_models.py --niters 300 -n 10000 -b 50 -l 15 --dataset hopper --latent-ode --rec-dims 30 -s 30 --gru-units 100 --units 300 --gen-layers 3 --rec-layers 3 --lr 0.01 
```

Physionet:
```
python3 run_models.py --niters 100 -n 8000 -b 100 -l 20 --dataset physionet --latent-ode --rec-dims 30 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --lr 0.01 --quantization 0.1 --classif
```


(+) Human Activity:
```
python3 run_models.py --niters 200 -n 10000 -b 100 -l 15 --dataset activity --latent-ode --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 500 --gru-units 50 --lr 0.01 --classif  --linear-classif
```


### Running different models

ODE-RNN:
```
python3 run_models.py --niters 500 -n 1000  -l 10  --lr 0.01 --dataset periodic  --ode-rnn
```

Latent ODE with ODE-RNN encoder:
```
python3 run_models.py --niters 500 -n 1000  -l 10  --lr 0.01 --dataset periodic  --latent-ode
```

Latent ODE with ODE-RNN encoder and poisson likelihood:
```
python3 run_models.py --niters 500 -n 1000  -l 10  --lr 0.01 --dataset periodic  --latent-ode --poisson
```

Latent ODE with RNN encoder (Chen et al, 2018):
```
python3 run_models.py --niters 500 -n 1000  -l 10  --lr 0.01 --dataset periodic  --latent-ode --z0-encoder rnn
```

RNN-VAE:
```
python3 run_models.py --niters 500 -n 1000   -l 10  --lr 0.01 --dataset periodic  --rnn-vae
```

Classic RNN:
```
python3 run_models.py --niters 500 -n 1000   -l 10  --lr 0.01 --dataset periodic  --classic-rnn
```

GRU-D
Consists of two parts: input imputation (--input-decay) and exponential decay of the hidden state (--rnn-cell expdecay)
```
python3 run_models.py --niters 500 -n 100  -b 30  -l 10 --lr 0.01 --dataset periodic  --classic-rnn --input-decay --rnn-cell expdecay
```




