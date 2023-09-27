# ZeroSwap
 
Our codebase has the following important files:

## main.py 

This file initializes the Glosten-Milgrom model and takes the informedness (alpha) and volatility (sigma) as inputs. It outputs a csv file which stores average modetary losses compared over different policies (Algorithms 1 and 2 in the paper, and the loss oracle algorithm) and plots of how the ask and bid price vary with time, along with plots of monetary loss vs time, mid-price deviation vs time and the spread vs time. By default, the performance of the bayesian policy and the oracle policy is also included in these. To run the file, use the command `python main.py --alpha <alpha> --sigma <sigma>`.

You may want to change the following variables in this file for running any experiments.
- `--max_episode_len` : Number of discrete time steps per sample path. Default value is 20000. 
- `--max_episodes` : Number of distinct sample paths to be averaged over. Default value is 120.

Other variables can be changed directly in the file itself. Their function has been indicated in the comments.

## run.sh

This file runs multiple simulations of `main.py`. 

## plot.py

This file plots the monetary loss comparison averaged over multiple simulations and different values of alpha and sigma. We recommend this file be run after `run.sh` completes.

## main_notebook.ipynb

Same as `main.py` but in a Jupyter notebook format. Also shows the output plots in the notebook in addition to storing them.

## env.py

This file contains all the code to initialize and run the Glosten-Milgrom model environment. Also implements the Bayesian market maker.

## agent.py

This file contains all the code for initializing and training RL agents on the Glosten-Milgrom environment.

## requirements.txt

The python packages used have been listed in the`requirements.txt`. They can be installed using the following command.

```
pip install -r requirements.txt
```

Alternatively, they can be installed using `conda` in a new environment called `zeroswap`.
```
conda create --name zeroswap python=3.8
conda activate zeroswap
conda install gym numpy matplotlib torch scipy pandas tqdm
```
