import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd

import gym
from gym.spaces import Discrete, Tuple, MultiDiscrete
import math
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm

from scipy import stats

from env import GlostenMilgromEnv, discretized_gaussian, get_args
from agent import DQN_Agent,QLearningAgent,QLearningAgentUpperConf,BayesianAgent, save_checkpoint, load_checkpoint

#_______________________________________SIMULATION_PARAMETERS_____________________________________________________________________________________________#

args = get_args()

print("Initializing GM model with params ALPHA = {0}, SIGMA = {1}".format(args.ALPHA,args.SIGMA))

# Define the initial true price, initial spread
p_ext = 1000
spread = 2

# Define other env params
mu = 18
spread_exp = 1 # spread penalty = - mu * (spread)^(spread_exp)

max_history_len = 21 # history over which imbalance calculated
max_episode_len = args.max_episode_len # number of time slots
max_episodes = args.max_episodes # number of episodes
average_over_episodes = True # Plot average of metric over episodes?
start_average_from = 20 # Take average in steady state after these manyn epsodes
ema_base = -1 #0.97 # exponential moving average - set to -1 if want to use moving window instead

informed = args.ALPHA # ALPHA : percentage of informed traders
vary_informed = False # true if alpha follows a random walk


jump_prob = args.SIGMA # SIGMA : probability of price jump for the model
vary_jump_prob = False # true if sigma follows a random walk
jump_size = 1 # size of price jump
jump_at = -100000 # = -1 if no jumps, if positive, then jumps at that time by 1000*jump_size and stays constant

use_short_term = True # use short term imbalance ?
use_endogynous = True # use endogynous variables or exogenous

n_price_adjustments = 3 # number of actions of agent to increase/decrease mid price

adjust_mid_spread = True # adjust mid + spread or adjust bid, ask separately
fixed_spread = False # fix the spread?
use_stored_path = True # use a stored sample path again?

compare = True # if true then compares with the loss oracle market maker
compare_with_bayes = True # if true then compares with the bayesian market maker

special_string = None
model_transfer = False # set to True if you are reusing the same agent - modify the special string above to indicate that

# Define type of the RL agent 
agent_type = "QT" # BA = Bayesian, QT = q learning using table, QUCB = q learning with UCB and table, DQN = deep q network, SARSA = sarsa dq

state_is_vec = False
# Define agent params
alpha = 0.06 # learning rate
gamma = 0.99 # discount rate of future rewards
epsilon = 0.99 # probability of exploration vs exploitation - decays over time, this is only the starting epsilon
c = 0.001 # UCB factor
if model_transfer:
    epsilon = 0.1
    
# Define plot params
moving_avg = int(max_episode_len/200)
if moving_avg < 10:
    moving_avg = 1
mode="valid"

# General trader models
noise_type = "Bernoulli" # "Bernoulli", "Gaussian", "Laplacian", "GeomGaussian"
noise_mean = 0
noise_variance = 1


#____________________________________________________________________________________________________________________________________#



# Create the environment
env = GlostenMilgromEnv(
    p_ext, 
    spread, 
    mu, 
    jump_prob=jump_prob,
    informed=informed, 
    max_episode_len=max_episode_len, 
    max_history_len=max_history_len,
    use_short_term=use_short_term,
    use_endogynous=use_endogynous,
    n_price_adjustments=n_price_adjustments,
    adjust_mid_spread=adjust_mid_spread,
    fixed_spread=fixed_spread,
    use_stored_path=use_stored_path,
    spread_exp=spread_exp,
    jump_size=jump_size,
    vary_informed=vary_informed,
    vary_jump_prob=vary_jump_prob,
    ema_base=ema_base,
    compare_with_bayes = compare_with_bayes,
    jump_at=jump_at,
    noise_type=noise_type,
    noise_variance=noise_variance,
)

env_compare = GlostenMilgromEnv(
    p_ext, 
    spread, 
    mu, 
    jump_prob=jump_prob,
    informed=informed, 
    max_episode_len=max_episode_len, 
    max_history_len=max_history_len,
    use_short_term=use_short_term,
    use_endogynous=not use_endogynous,
    n_price_adjustments=n_price_adjustments,
    adjust_mid_spread=adjust_mid_spread,
    fixed_spread=fixed_spread,
    use_stored_path=use_stored_path,
    spread_exp=spread_exp,
    jump_size=jump_size,
    vary_informed=vary_informed,
    vary_jump_prob=vary_jump_prob,
    ema_base=ema_base,
    compare_with_bayes = compare_with_bayes,
    jump_at=jump_at,
    noise_type=noise_type,
    noise_variance=noise_variance,
)

env_bayes = GlostenMilgromEnv(
    p_ext, 
    spread, 
    mu, 
    jump_prob=jump_prob,
    informed=informed, 
    max_episode_len=max_episode_len, 
    max_history_len=max_history_len,
    use_short_term=use_short_term,
    use_endogynous=use_endogynous,
    n_price_adjustments=n_price_adjustments,
    adjust_mid_spread=adjust_mid_spread,
    fixed_spread=fixed_spread,
    use_stored_path=use_stored_path,
    spread_exp=spread_exp,
    jump_size=jump_size,
    vary_informed=vary_informed,
    vary_jump_prob=vary_jump_prob,
    ema_base=ema_base,
    compare_with_bayes = compare_with_bayes,
    jump_at=jump_at,
    noise_type=noise_type,
    noise_variance=noise_variance,
)
# Create the agent


if model_transfer:
    if agent_type == "DQN" or agent_type == "PPO":
        state_is_vec = True
else:
    if agent_type == "QT": # tabular q learning with epsilon exploration
        n_states = 2*max_history_len + 1  # Define the number of discrete states for the given history window
        agent = QLearningAgent(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            epsilon=epsilon
        )
        comparison_agent = QLearningAgent(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            epsilon=epsilon
        )
    elif agent_type == "QUCB": # tabular q learning + ucb exploration
        n_states = 2*max_history_len + 1  # Define the number of discrete states for the given history window
        agent = QLearningAgentUpperConf(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            c=c
        )
        comparison_agent = QLearningAgentUpperConf(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            c=c
        )
    elif agent_type == "DQN":
        state_dim = 1
        state_is_vec = True
        agent = DQN_Agent(
            max_history_len,
            n_price_adjustments,
            num_adjustments=n_price_adjustments,
            window=max_history_len,
            hidden_size=64,
            lr=1e-3,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
    elif agent_type == "SARSA":
        state_dim = 1
        agent = SARSA_agent(
            n_actions=[env.action_space[0].n,env.action_space[1].n],
            state_dim=state_dim,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
    elif agent_type == "UCRL":
        pass
    elif agent_type == "TD":
        pass
    elif agent_type == "AI":
        pass
    elif agent_type == "BA":
        n_states = 2*max_history_len + 1
        agent = BayesianAgent(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            epsilon=epsilon
        )
    else:
        print("ERROR_UNKNOWN_AGENT_TYPE")

if compare_with_bayes:
    n_states_bayes = 2*max_history_len + 1
    bayesian_agent = BayesianAgent(
        n_actions=[env.action_space[0].n,env.action_space[1].n], 
        n_states=n_states_bayes, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon
    )



import os

# Train the agent for some number of episodes - ideally should only need one episode for training the network
# output detailed plots for the last episode if average_over_episodes is false

total_rewards = []
monetary_losses = []

total_rewards_compare = []
monetary_losses_compare = []

rewards_vs_time = [0.0]
monetary_losses_vs_time = [0.0]
spread_vs_time = []
ask_vs_time = []
bid_vs_time = []
mid_price_vs_time = []

rewards_vs_time_compare = [0.0]
monetary_losses_vs_time_compare = [0.0]
spread_vs_time_compare = []
ask_vs_time_compare = []
bid_vs_time_compare = []
mid_price_vs_time_compare = []

total_rewards_bayes = []
monetary_losses_bayes = []
rewards_vs_time_bayes = [0.0]
monetary_losses_vs_time_bayes = [0.0]
spread_vs_time_bayes = []
ask_vs_time_bayes = []
bid_vs_time_bayes = []
mid_price_vs_time_bayes = []


p_ext_vs_time = []
price_of_ask_over_time = []

def normalize_trade_imbalance(imbalance, n_states):
    return min(max(int(imbalance + n_states // 2), 0), n_states - 1)


if not state_is_vec:
    n_states = agent.n_states

    
for episode in tqdm(range(max_episodes)):
    
    #print("Episode ",episode)
    env.reset()
    env.resetAllVars(reset_stored_path=True)
    
    env_compare.reset()
    env_compare.resetAllVars(reset_stored_path=False)
    
    env_bayes.reset()
    env_bayes.resetAllVars(reset_stored_path=False)
    
    env_compare.price_path = env.price_path
    env_bayes.price_path = env.price_path
    
    env_compare.trader_price_path = env.trader_price_path
    env_bayes.trader_price_path = env.trader_price_path
    
    if not state_is_vec:
        state = normalize_trade_imbalance(env.imbalance, n_states)
        state_compare = normalize_trade_imbalance(env_compare.imbalance, n_states)
    else:
        state = torch.zeros(1,max_history_len)
        state_compare = torch.zeros(1,max_history_len)
    
    state_bayes = normalize_trade_imbalance(env_bayes.imbalance, n_states)
    
    done = False
    
    total_reward = 0
    total_reward_compare = 0
    total_reward_bayes = 0
    
    time = 0

    while not done:
        action = agent.choose_action(state,epsilon=agent.epsilon**time)
        
        (next_trade_history, next_imbalance), reward, done, extra_dict = env.step(action)
        
        if state_is_vec:
            next_state = torch.tensor(next_trade_history).permute((1,0)).float()
            agent.update(state, action, reward, next_state, done=done)
        else:
            next_state = normalize_trade_imbalance(next_imbalance, n_states)
            agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        time += 1
        
        if not average_over_episodes:
            if episode == max_episodes-1:
                p_ext_vs_time.append(extra_dict["p_ext"])
                rewards_vs_time.append(reward)
                monetary_losses_vs_time.append(extra_dict["monetary_loss"])
                spread_vs_time.append(extra_dict["spread"])
                ask_vs_time.append(extra_dict["ask"])
                bid_vs_time.append(extra_dict["bid"])
                mid_price_vs_time.append(extra_dict["mid"])
                #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
        else:
            if episode == start_average_from:
                p_ext_vs_time.append(extra_dict["p_ext"])
                rewards_vs_time.append(reward)
                monetary_losses_vs_time.append(extra_dict["monetary_loss"])
                spread_vs_time.append(extra_dict["spread"])
                ask_vs_time.append(extra_dict["ask"])
                bid_vs_time.append(extra_dict["bid"])
                mid_price_vs_time.append(extra_dict["mid"])
                #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
            elif episode > start_average_from:
                p_ext_vs_time[time-1] += (extra_dict["p_ext"])
                rewards_vs_time[time-1] +=(reward)
                monetary_losses_vs_time[time-1] +=(extra_dict["monetary_loss"])
                spread_vs_time[time-1] +=(extra_dict["spread"])
                ask_vs_time[time-1] +=(extra_dict["ask"])
                bid_vs_time[time-1] +=(extra_dict["bid"])
                mid_price_vs_time[time-1] +=(extra_dict["mid"])
            if episode == max_episodes-1:
                p_ext_vs_time[time-1] /= max_episodes
                rewards_vs_time[time-1] /= max_episodes
                monetary_losses_vs_time[time-1] /= max_episodes
                spread_vs_time[time-1] /= max_episodes
                ask_vs_time[time-1] /= max_episodes
                bid_vs_time[time-1] /= max_episodes
                mid_price_vs_time[time-1] /= max_episodes
    
        if compare:
            action = comparison_agent.choose_action(state_compare,epsilon=epsilon**time)

            (next_trade_history, next_imbalance), reward, done, extra_dict = env_compare.step(action)

            next_state_compare = normalize_trade_imbalance(next_imbalance, n_states)

            comparison_agent.update(state_compare, action, reward, next_state_compare)

            state_compare = next_state_compare
            total_reward_compare += reward

            if not average_over_episodes:
                if episode == max_episodes-1:
                    #p_ext_vs_time_compare.append(extra_dict["p_ext"])
                    rewards_vs_time_compare.append(reward)
                    monetary_losses_vs_time_compare.append(extra_dict["monetary_loss"])
                    spread_vs_time_compare.append(extra_dict["spread"])
                    ask_vs_time_compare.append(extra_dict["ask"])
                    bid_vs_time_compare.append(extra_dict["bid"])
                    mid_price_vs_time_compare.append(extra_dict["mid"])
                    #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
            else:
                if episode == start_average_from:
                    #p_ext_vs_time_compare.append(extra_dict["p_ext"])
                    rewards_vs_time_compare.append(reward)
                    monetary_losses_vs_time_compare.append(extra_dict["monetary_loss"])
                    spread_vs_time_compare.append(extra_dict["spread"])
                    ask_vs_time_compare.append(extra_dict["ask"])
                    bid_vs_time_compare.append(extra_dict["bid"])
                    mid_price_vs_time_compare.append(extra_dict["mid"])
                    #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
                elif episode > start_average_from:
                    #p_ext_vs_time_compare[time-1] += (extra_dict["p_ext"])
                    rewards_vs_time_compare[time-1] +=(reward)
                    monetary_losses_vs_time_compare[time-1] +=(extra_dict["monetary_loss"])
                    spread_vs_time_compare[time-1] +=(extra_dict["spread"])
                    ask_vs_time_compare[time-1] +=(extra_dict["ask"])
                    bid_vs_time_compare[time-1] +=(extra_dict["bid"])
                    mid_price_vs_time_compare[time-1] +=(extra_dict["mid"])
                if episode == max_episodes-1:
                    #p_ext_vs_time_compare[time-1] /= max_episodes
                    rewards_vs_time_compare[time-1] /= max_episodes
                    monetary_losses_vs_time_compare[time-1] /= max_episodes
                    spread_vs_time_compare[time-1] /= max_episodes
                    ask_vs_time_compare[time-1] /= max_episodes
                    bid_vs_time_compare[time-1] /= max_episodes
                    mid_price_vs_time_compare[time-1] /= max_episodes
        if compare_with_bayes:              
            action = bayesian_agent.choose_action(state_bayes,epsilon=epsilon**time)
            
            (next_trade_history, next_imbalance), reward, done, extra_dict = env_bayes.step(action)

            next_state_bayes = normalize_trade_imbalance(next_imbalance, n_states)

            bayesian_agent.update(state_bayes, action, reward, next_state_bayes)

            state_bayes = next_state_bayes
            total_reward_bayes += reward

            if not average_over_episodes:
                if episode == max_episodes-1:
                    #p_ext_vs_time_bayes.append(extra_dict["p_ext"])
                    rewards_vs_time_bayes.append(reward)
                    monetary_losses_vs_time_bayes.append(extra_dict["monetary_loss"])
                    spread_vs_time_bayes.append(extra_dict["spread"])
                    ask_vs_time_bayes.append(extra_dict["ask"])
                    bid_vs_time_bayes.append(extra_dict["bid"])
                    mid_price_vs_time_bayes.append(extra_dict["mid"])
                    #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
            else:
                if episode == start_average_from:
                    #p_ext_vs_time_bayes.append(extra_dict["p_ext"])
                    rewards_vs_time_bayes.append(reward)
                    monetary_losses_vs_time_bayes.append(extra_dict["monetary_loss"])
                    spread_vs_time_bayes.append(extra_dict["spread"])
                    ask_vs_time_bayes.append(extra_dict["ask"])
                    bid_vs_time_bayes.append(extra_dict["bid"])
                    mid_price_vs_time_bayes.append(extra_dict["mid"])
                    #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
                elif episode > start_average_from:
                    #p_ext_vs_time_bayes[time-1] += (extra_dict["p_ext"])
                    rewards_vs_time_bayes[time-1] +=(reward)
                    monetary_losses_vs_time_bayes[time-1] +=(extra_dict["monetary_loss"])
                    spread_vs_time_bayes[time-1] +=(extra_dict["spread"])
                    ask_vs_time_bayes[time-1] +=(extra_dict["ask"])
                    bid_vs_time_bayes[time-1] +=(extra_dict["bid"])
                    mid_price_vs_time_bayes[time-1] +=(extra_dict["mid"])
                if episode == max_episodes-1:
                    #p_ext_vs_time_bayes[time-1] /= max_episodes
                    rewards_vs_time_bayes[time-1] /= max_episodes
                    monetary_losses_vs_time_bayes[time-1] /= max_episodes
                    spread_vs_time_bayes[time-1] /= max_episodes
                    ask_vs_time_bayes[time-1] /= max_episodes
                    bid_vs_time_bayes[time-1] /= max_episodes
                    mid_price_vs_time_bayes[time-1] /= max_episodes
                        
    total_rewards.append(total_reward)
    monetary_losses.append(env.cumulative_monetary_loss)
    if compare:
        total_rewards_compare.append(total_reward_compare)
        monetary_losses_compare.append(env_compare.cumulative_monetary_loss)
    if compare_with_bayes:
        total_rewards_bayes.append(total_reward_bayes)
        monetary_losses_bayes.append(env_bayes.cumulative_monetary_loss)






#______________________________________________________________________________________________________________________________#

if noise_type == "Bernoulli":
    figure_path = "modelFreeGM/informed_{0}_jump_{1}_mu_{2}/fixedSpread_{10}_useShortTerm_{3}_useEndo_{4}_maxHistoryLen_{5}/agentType_{6}_alpha_{7}_gamma_{8}_epsilon_{9}".format(
        informed,
        jump_prob,
        mu,
        use_short_term,
        use_endogynous,
        max_history_len,
        agent_type,
        alpha,
        gamma,
        epsilon,
        fixed_spread
    )
else:
    figure_path = "modelFreeGM/{0}_{10}_jump_{1}_mu_{2}/fixedSpread_{10}_useShortTerm_{3}_useEndo_{4}_maxHistoryLen_{5}/agentType_{6}_alpha_{7}_gamma_{8}_epsilon_{9}".format(
        noise_type,
        jump_prob,
        mu,
        use_short_term,
        use_endogynous,
        max_history_len,
        agent_type,
        alpha,
        gamma,
        epsilon,
        fixed_spread,
        noise_variance
    ) 
if special_string is not None:
    figure_path = figure_path + "/{0}".format(special_string)

if ema_base != -1:
    figure_path = figure_path + "/ema_base_{0}".format(ema_base)
    
if not adjust_mid_spread:
    figure_path = figure_path + "/direct_ask_bid_control"

os.makedirs(figure_path , exist_ok=True)

# Calculate the average total reward and monetary loss
average_total_reward = np.mean(total_rewards)
average_monetary_loss = np.mean(monetary_losses)
print("Average total reward main:", average_total_reward/max_episode_len)
print("Average monetary loss main:", average_monetary_loss/max_episode_len)
print("Mean spread main",np.mean(np.array(spread_vs_time)))
print("Mean mid dev main",np.mean(abs(np.array(mid_price_vs_time)-np.array(p_ext_vs_time))))
print(" ")
if compare:
    average_total_reward_compare = np.mean(total_rewards_compare)
    average_monetary_loss_compare = np.mean(monetary_losses_compare)
    print("Average total reward compare:", average_total_reward_compare/max_episode_len)
    print("Average monetary loss compare:", average_monetary_loss_compare/max_episode_len)
    print("Mean spread reference",np.mean(np.array(spread_vs_time_bayes)))
    print("Mean mid dev reference",np.mean(abs(np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))))
    print(" ")
if compare_with_bayes:
    average_total_reward_bayes = np.mean(total_rewards_bayes)
    average_monetary_loss_bayes = np.mean(monetary_losses_bayes)
    print("Average total reward reference:", average_total_reward_bayes/max_episode_len)
    print("Average monetary loss reference:", average_monetary_loss_bayes/max_episode_len)
    print("Mean spread reference",np.mean(np.array(spread_vs_time_bayes)))
    print("Mean mid dev reference",np.mean(abs(np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))))
    print(" ")

import csv

with open('losses_and_spreads_modified.csv', 'a', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow([
    env.informed,
    env.jump_prob,
    " ",
    np.mean(np.array(monetary_losses)[start_average_from:]/max_episode_len),
    np.median(np.array(monetary_losses)[start_average_from:]/max_episode_len),
    np.mean(np.array(spread_vs_time)),
    np.median(np.array(spread_vs_time)),
    np.mean(abs(np.array(mid_price_vs_time)-np.array(p_ext_vs_time))),
    np.median(abs(np.array(mid_price_vs_time)-np.array(p_ext_vs_time))),
    " ",
    np.mean(np.array(monetary_losses_compare)[start_average_from:]/max_episode_len),
    np.median(np.array(monetary_losses_compare)[start_average_from:]/max_episode_len),
    np.mean(np.array(spread_vs_time_compare)),
    np.median(np.array(spread_vs_time_compare)),
    np.mean(abs(np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))),
    np.median(abs(np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))),
    " ",
    np.mean(np.array(monetary_losses_bayes)[start_average_from:]/max_episode_len),
    np.median(np.array(monetary_losses_bayes)[start_average_from:]/max_episode_len),
    np.mean(np.array(spread_vs_time_bayes)),
    np.median(np.array(spread_vs_time_bayes)),
    np.mean(abs(np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))),
    np.median(abs(np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))),
    ])

# Plot the average monetary loss over all episodes
plt.plot(monetary_losses)
if compare:
    plt.plot(monetary_losses_compare)
if compare_with_bayes:
    plt.plot(monetary_losses_bayes)
plt.xlabel("Episode Number")
plt.ylabel("Average Monetary Loss vs time")
plt.title("Average Monetary Loss over Episodes")
plt.legend()
filename = "Monetary_Loss.pdf"
file_path = os.path.join(figure_path, filename)
plt.savefig(file_path)  
plt.close()
 

# Plot the average total reward over all episodes
plt.plot(total_rewards)
if compare:
    plt.plot(total_rewards_compare)
if compare_with_bayes:
    plt.plot(total_rewards_bayes)
plt.xlabel("Episode Number")
plt.ylabel("Average total reward")
plt.title("Average total reward over Episodes")
plt.legend()
filename = "Total_Reward.pdf"
file_path = os.path.join(figure_path, filename)
plt.savefig(file_path)
plt.close()



# Plot the monetary loss over time for the last episode
plt.plot(np.convolve(np.array(monetary_losses_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="Main")
if compare:
    plt.plot(np.convolve(np.array(monetary_losses_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")
if compare_with_bayes:
    plt.plot(np.convolve(np.array(monetary_losses_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode),label="Reference")
plt.xlabel("Time")
plt.ylabel("Monetary Loss")
plt.title("Monetary Loss over time")
plt.legend()
filename = "Loss_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
plt.savefig(file_path)
plt.close()


# Plot the reward over time for the last episode
plt.plot(np.convolve(np.array(rewards_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="orig")
if compare:
    plt.plot(np.convolve(np.array(rewards_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")
plt.xlabel("Time")
plt.ylabel("Reward")#
plt.title("Reward over time")
plt.legend()
filename = "Reward_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
plt.savefig(file_path)
plt.close()


# Plot the spread over time for the last episode
fig, ax = plt.subplots()

ax.plot(np.convolve(np.array(spread_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="Q-learning")# spread should decay with time
# if compare:
#     plt.plot(np.convolve(np.array(spread_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")# spread should decay with time
if compare_with_bayes:
    ax.plot(np.convolve(np.array(spread_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode),label="Bayesian")# spread should decay with time
if compare:
    ax.plot(np.convolve(np.array(spread_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="Bayesian")# spread should decay with time

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Spread', fontsize=14)
ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
#ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing

print("Spread over time : one sample path")
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

filename = "Spread_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
fig.savefig(file_path, format='pdf', bbox_inches='tight')
plt.close()

# Plot the spread distribution for the last episode
pd.Series(spread_vs_time).hist()
# if compare:
#     plt.plot(np.convolve(np.array(spread_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")# spread should decay with time
# if compare_with_bayes:
#     plt.plot(np.convolve(np.array(spread_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode),label="Bayes")# spread should decay with time
# plt.xlabel("Time")
# plt.ylabel("Spread")
# plt.title("Spread over time : one sample path")
# plt.legend()
# filename = "Spread_Vs_time.pdf"
# file_path = os.path.join(figure_path, filename)
# plt.savefig(file_path)
plt.close()

# Plot the ask,bid and external price over time for the last episode
fig, ax = plt.subplots()

ax.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{ext}$")
ax.plot(np.convolve(np.array(ask_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{ask}$")
ax.plot(np.convolve(np.array(bid_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{bid}$")

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Price', fontsize=14)

ax.legend()

ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
#ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

print("Q-learning Ask and Bid over time : one sample path")  # Adjust the title and fontsize as required

filename = "AskBid_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
fig.savefig(file_path, format='pdf', bbox_inches='tight')

plt.close()

if compare:
    plt.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="P_ext")
    plt.plot(np.convolve(np.array(ask_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode), label="Ask")
    plt.plot(np.convolve(np.array(bid_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode), label="Bid")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Compare Ask and Bid over time : one sample path (reward has p_ext)")
    plt.legend()
    filename = "AskBid_Vs_time_compare.pdf"
    file_path = os.path.join(figure_path, filename)
    plt.savefig(file_path)
    plt.close()
if compare_with_bayes:
    fig, ax = plt.subplots()

    ax.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{ext}$")
    ax.plot(np.convolve(np.array(ask_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{ask}$")
    ax.plot(np.convolve(np.array(bid_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{bid}$")
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    
    #ax.title("Reference Ask and Bid over time : one sample path")
    ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    filename = "AskBid_Vs_time_bayes.pdf"
    file_path = os.path.join(figure_path, filename)
    plt.savefig(file_path)
    plt.close()

# Plot the mid and ext price over time for the last episode
fig, ax = plt.subplots()

ax.plot(abs(np.convolve((np.array(mid_price_vs_time)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode)), label="Q-learning")
if compare_with_bayes:
    ax.plot(abs(np.convolve((np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode)), label="Bayesian")# need to make sure E[(mid-p_ext)^2]-> 0 as t becomes large for the last episode
if compare:
    ax.plot(abs(np.convolve((np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode)), label="Bayesian")# need to make sure E[(mid-p_ext)^2]-> 0 as t becomes large for the last episode

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Mid Price Deviation', fontsize=14)

ax.legend()

ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
#ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

print("mid price deviation")  # Adjust the title and fontsize as required

filename = "Mid_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
fig.savefig(file_path, format='pdf', bbox_inches='tight')

plt.close()

fig, ax = plt.subplots()
if env.vary_jump_prob:
    ax.plot(np.convolve(np.array(env.jump_prob_path),np.ones(moving_avg)/moving_avg,mode=mode), label="Volatility")
if env.vary_informed:
    ax.plot(np.convolve(np.array(env.informed_path),np.ones(moving_avg)/moving_avg,mode=mode), label="Informedness")

if env.vary_informed or env.vary_jump_prob:
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Parameter Value', fontsize=14)
    ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing

    #plt.yscale("log")
    print("Variable volatility and informedness")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    filename = "Params_Vs_time.pdf"
    file_path = os.path.join(figure_path, filename)
    fig.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.close()




