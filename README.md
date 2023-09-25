# ZeroSwap
 
Our codebase has the following important files:

## main.py 

This file initializes the Glosten-Milgrom model and takes as input the informedness (alpha) and volatility (sigma) as inputs. It outputs a csv file which stores average modetary losses compared over different policies and plots of how the ask and bid price vary with time, along with plots of monetary loss vs time, mid-price deviation vs time and the spread vs time. By default, the performance of the bayesian policy and the oracle policy is also included in these. To run the file, use the command `python main.py --alpha <alpha> --sigma <sigma>`

## run.sh

This file runs multiple simulations of `main.py`. 

## plot.py

This file plots the monetary loss comparison averaged over multiple simulations and different values of alpha and sigma. We recommend this file be run after `run.sh` completes.

## main_notebook.ipynb

Same `main.py` but in a Jupyter notebook format. ALso shows the output plots in the notebook in addition to storing them.

## env.py

This file contains all the code to initialize and run the Glosten-Milgrom model environment. Also implements the Bayesian market maker.

## agent.py

This file contains all the code for initializing and training RL agents on the Glosten-Milgrom environment.