import gym
import numpy as np
from gym.spaces import Discrete, Tuple, MultiDiscrete
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys,csv,math
import numpy as np
import scipy.stats as stats
import os
import pandas as pd

from agent import CFMM

import argparse



def discretized_gaussian(length, variance):
    # Calculate the mean and standard deviation
    mean = length / 2
    std_dev = np.sqrt(variance)

    # Create an array from 0 to length
    x = np.arange(length)

    # Create a normal distribution with the calculated mean and standard deviation
    distribution = stats.norm.pdf(x, mean, std_dev)

    # Normalize the distribution so it sums to 1 (as a discrete probability distribution should)
    distribution /= distribution.sum()

    return distribution



class GlostenMilgromEnv(gym.Env):
    def __init__(
        self,
        p_ext,
        spread,
        mu,
        jump_prob=0.5,
        jump_variance=16,
        informed=0.5,
        max_episode_len=100,
        max_history_len=10,
        use_short_term=True,
        use_endogynous=True,
        n_price_adjustments=21,
        adjust_mid_spread=False,
        fixed_spread=False,
        use_stored_path=False,
        spread_exp=2,
        jump_size=1,
        vary_informed=False,
        vary_jump_prob=False,
        ema_base=-1,
        compare_with_bayes=False,
        jump_at=-1,
        noise_type="Bernoulli",
        noise_variance=1,
        n_slippage_adjustments=3,
        max_trade_size=1,
        slippage=0.1,
        normal_AMM=False
    ):
        super(GlostenMilgromEnv, self).__init__()
        self.p_ext = p_ext
        self.current_p_ext = self.p_ext
        self.spread = spread
        self.mu = mu
        self.jump_prob = jump_prob
        self.jump_variance = jump_variance
        self.informed = informed
        self.use_short_term = use_short_term
        self.use_endogynous = use_endogynous
        self.n_price_adjustments = n_price_adjustments
        self.adjust_mid_spread = adjust_mid_spread
        self.fixed_spread = fixed_spread
        self.use_stored_path = use_stored_path
        self.spread_exp = spread_exp
        self.jump_size = jump_size
        self.vary_informed = vary_informed
        self.vary_jump_prob = vary_jump_prob
        self.ema_base = ema_base
        self.compare_with_bayes = compare_with_bayes
        self.noise_type = noise_type
        self.noise_variance = noise_variance
        self.jump_at = jump_at
        
        self.time_step = 0
        self.max_history_len = max_history_len
        self.max_episode_len = max_episode_len
        self.trade_history = np.zeros((self.max_history_len, 1))
        self.imbalance = 0
        self.short_term_imbalance = 0
        self.short_term = max_history_len
        self.bid_price = self.current_p_ext - spread / 2
        self.ask_price = self.current_p_ext + spread / 2
        self.mid = self.current_p_ext
        self.max_trade_size = max_trade_size
        self.slippage = slippage
        self.n_slippage_adjustments = n_slippage_adjustments
        self.cumulative_monetary_loss = 0
        self.max_episode_len = max_episode_len
        self.informed_arr = []
        self.jump_prob_arr = []
        
        self.bayesian_ask = self.current_p_ext + spread / 2
        self.bayesian_bid = self.current_p_ext - spread / 2
        self.belief = np.zeros(2001)
        self.belief_center = self.current_p_ext
        self.belief[1000] = 1
        self.belief_vals = np.arange(len(self.belief))-(len(self.belief)-1)/2+self.belief_center
        self.noise_pdf = discretized_gaussian(200,noise_variance)

        x_0 = 100000
        self.bayesian_AMM = CFMM([x_0,self.p_ext*x_0],"mean",[0.5],0)
        self.normal_AMM = normal_AMM
        
        self.generate_path(jump_at=self.jump_at)

        if self.adjust_mid_spread:
            self.action_space = MultiDiscrete([self.n_price_adjustments, 3])  # Two discrete variables: mid and spread price increments
        else:
            self.action_space = MultiDiscrete([self.n_price_adjustments, self.n_price_adjustments])  # Two discrete variables: bid and ask price increments
        
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=2, shape=(self.max_history_len, 1), dtype=np.uint8),
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        ))
        
    def resetAllVars(
        self,
        reset_stored_path = True
    ):
        super(GlostenMilgromEnv, self).__init__()
        #self.p_ext = p_ext
        self.current_p_ext = self.p_ext
        self.spread = 2
        
        self.time_step = 0
        self.trade_history = np.zeros((self.max_history_len, 1))
        self.imbalance = 0
        self.short_term_imbalance = 0
        self.short_term = self.max_history_len
        self.bid_price = self.current_p_ext - self.spread / 2
        self.ask_price = self.current_p_ext + self.spread / 2
        self.mid = self.current_p_ext
        self.cumulative_monetary_loss = 0
        self.max_episode_len = self.max_episode_len
        
        self.bayesian_ask = self.current_p_ext
        self.bayesian_bid = self.current_p_ext
        self.belief = np.zeros(2001)
        self.belief_center = self.current_p_ext
        self.belief[1000] = 1
        self.belief_vals = np.arange(len(self.belief))-(len(self.belief)-1)/2+self.belief_center

        x_0 = 100000
        self.bayesian_AMM = CFMM([x_0,self.p_ext*x_0],"mean",[0.5],0)
                
        if reset_stored_path:
            self.generate_path(jump_at=self.jump_at)

        if self.adjust_mid_spread:
            self.action_space = MultiDiscrete([self.n_price_adjustments, 3])  # Two discrete variables: mid and spread price increments
        else:
            self.action_space = MultiDiscrete([self.n_price_adjustments, self.n_price_adjustments])  # Two discrete variables: bid and ask price increments
        
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=2, shape=(self.max_history_len, 1), dtype=np.uint8),
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        ))
    
    def generate_path(self, jump_at=-1):
        buffer = 10
        next_p_ext = self.current_p_ext
        self.price_path = []
        self.trader_price_path = []
        self.jump_prob_path = []
        self.informed_path = []
        jump_size = self.jump_size

        
        for t in range(self.max_episode_len+buffer):
            if t==jump_at:
                next_p_ext += 1000*jump_size
                #print(jump_at, "----------------------------------------------------")
                jump_size=0
            #print(jump_at)
            if self.vary_jump_prob:
                self.jump_prob += np.random.choice([-0.001, 0.001])
                self.jump_prob = np.clip(self.jump_prob, 0.01, 0.99)

            jump = np.random.choice(["jump up","jump down", "no jump"], p=[self.jump_prob/2, self.jump_prob/2, 1-self.jump_prob])
            if jump == "jump up":
                next_p_ext += jump_size
            if jump == "jump down":
                next_p_ext -= jump_size
            self.price_path.append(next_p_ext)
            self.jump_prob_path.append(self.jump_prob)
            
        if self.noise_type == "Bernoulli":
            
            for t in range(self.max_episode_len+buffer):
                if self.vary_informed:
                    self.informed += np.random.choice([-0.001, 0.001])
                    self.informed = np.clip(self.informed, 0.01, 0.99)

                trader_type = np.random.choice([True, False],  p=[self.informed, 1-self.informed])
                if trader_type : # informed trader
                    trader_price = self.price_path[t]
                else:
                    trader_price = np.random.choice([np.inf, -np.inf])
                self.trader_price_path.append(trader_price)
                self.informed_path.append(self.informed)

        elif self.noise_type == "Gaussian":
            self.trader_price_path = np.array(self.price_path)+np.random.normal(0,self.noise_variance,size=(self.max_episode_len+buffer))    
        elif self.noise_type == "Laplacian":
            self.trader_price_path = np.array(self.price_path)+np.random.laplace(0 ,math.sqrt(self.noise_variance/2),size=(self.max_episode_len+buffer))
        elif self.noise_type == "GeomGaussian":
            self.trader_price_path = np.array(self.price_path)+np.random.lognormal(0 ,self.noise_variance,size=(self.max_episode_len+buffer))
        
            
    def get_noisy_measurement_of_price(self):
        if self.use_stored_path:
            trader_price = self.trader_price_path[self.time_step]
        elif self.noise_type == "Bernoulli":
            trader_type = np.random.choice(["informed", "uninformed"], p=[self.informed, 1-self.informed])
            if trader_type == "informed":
                trader_price = self.current_p_ext
            else:
                trader_price = np.random.choice([np.inf, -np.inf])    
        elif self.noise_type == "Gaussian":
            trader_price = np.random.normal(self.current_p_ext,math.sqrt(self.noise_variance)) 
        elif self.noise_type == "Laplacian":
            trader_price = np.random.laplace(self.current_p_ext ,math.sqrt(self.noise_variance/2))
        elif self.noise_type == "GeomGaussian":
            trader_price = np.random.lognormal(self.current_p_ext ,self.noise_variance)
        
        return trader_price
            
    def get_noisy_measurement_of_price(self,variable=False):
        if self.use_stored_path:
            if variable:
                trader_price = self.trader_price_path_variable[self.time_step]
            else:
                trader_price = self.trader_price_path[self.time_step]
        elif self.noise_type == "Bernoulli":
            trader_type = np.random.choice(["informed", "uninformed"], p=[self.informed, 1-self.informed])
            if trader_type == "informed":
                trader_price = self.current_p_ext
            else:
                trader_price = np.random.choice([np.inf, -np.inf])    
        elif self.noise_type == "Gaussian":
            trader_price = np.random.normal(self.current_p_ext,math.sqrt(self.noise_variance)) 
        elif self.noise_type == "Laplacian":
            trader_price = np.random.laplace(self.current_p_ext ,math.sqrt(self.noise_variance/2))
        elif self.noise_type == "GeomGaussian":
            trader_price = np.random.lognormal(self.current_p_ext ,self.noise_variance)
        
        return trader_price
    
    def askBelief(self,ask):
        ask_idx = int(ask - self.belief_center + (len(self.belief)-1)/2)
        belief = self.belief.copy()
        belief[ask_idx:] = belief[ask_idx:] * (1+self.informed)
        belief[0:ask_idx] = belief[0:ask_idx] * (1-self.informed)
        belief = belief/np.sum(belief)
        return belief
        
    def bidBelief(self,bid):
        bid_idx = int(bid - self.belief_center + (len(self.belief)-1)/2)
        belief = self.belief.copy()
        belief[bid_idx:] = belief[bid_idx:] * (1-self.informed)
        belief[0:bid_idx] = belief[0:bid_idx] * (1+self.informed)
        belief = belief/np.sum(belief)
        return belief
        
    def bayesian_belief_update(self,update_type,variable=False,unknown=False):
        if variable:
            if unknown:
                pass
            else:
                jump_distribution = discretized_gaussian(100,self.jump_variance)
                convolved_vec = np.convolve(self.belief,jump_distribution,'same')
                self.belief = convolved_vec/np.sum(convolved_vec)
        else:
            if update_type == "price jump":
                #self.belief = (1-self.informed)*self.belief + self.informed*np.append(0.,self.belief[:-1])/2 + self.informed*np.append(self.belief[1:],0.)/2
                self.belief = (1-self.jump_prob)*self.belief + np.append(0.,self.belief[:-1])*self.jump_prob/2 + np.append(self.belief[1:],0.)*self.jump_prob/2
                self.belief = self.belief/np.sum(self.belief)
            elif update_type == "trader buy action":
                self.belief = self.askBelief(self.bayesian_ask)
            elif update_type == "trader sell action":
                self.belief = self.bidBelief(self.bayesian_bid)
            elif update_type == "trader null action":
                ask_idx = int(self.bayesian_ask - self.belief_center + (len(self.belief)-1)/2)
                bid_idx = int(self.bayesian_bid - self.belief_center + (len(self.belief)-1)/2)
                self.belief[0:bid_idx] = self.belief[0:bid_idx]*(1-self.informed)
                self.belief[ask_idx+1:] = self.belief[ask_idx+1:]*(1-self.informed)
                self.belief = self.belief/np.sum(self.belief)
        
    def get_bayesian_ask(self):
        lhs = self.bayesian_ask
        rhs = np.dot(self.askBelief(lhs),self.belief_vals)
        initial_sign = (lhs > rhs)
        final_sign = initial_sign.copy()
        prevlhs = lhs
        prevrhs = rhs
        
        while final_sign == initial_sign:
            prevlhs = lhs
            prevrhs = rhs
            if lhs == self.belief_center-(len(self.belief)-1)/2 or lhs == self.belief_center+(len(self.belief)-1)/2:
                break
            if final_sign:
                lhs -= 1
            else:
                lhs += 1
            rhs = np.dot(self.askBelief(lhs),self.belief_vals)
            final_sign = (lhs > rhs)
        if abs(lhs-rhs) > abs(prevlhs-prevrhs):
            lhs = prevlhs
        return lhs
        
    def get_bayesian_bid(self):
        lhs = self.bayesian_bid
        rhs = np.dot(self.bidBelief(lhs),self.belief_vals)
        initial_sign = (lhs > rhs)
        final_sign = initial_sign.copy()
        prevlhs = lhs
        prevrhs = rhs
        
        while final_sign == initial_sign:
            prevlhs = lhs
            prevrhs = rhs
            if lhs == self.belief_center-(len(self.belief)-1)/2 or lhs == self.belief_center+(len(self.belief)-1)/2:
                break
            if final_sign:
                lhs -= 1
            else:
                lhs += 1
            rhs = np.dot(self.bidBelief(lhs),self.belief_vals)
            final_sign = (lhs > rhs)
        if abs(lhs-rhs) > abs(prevlhs-prevrhs):
            lhs = prevlhs
        return lhs
    
    def step(self, action):
        extra_dict = {}
        
        reward = 0
        if action == 0:
            self.bid_price = self.get_bayesian_bid()
            self.ask_price = self.get_bayesian_ask()
            self.bayesian_bid = self.bid_price
            self.bayesian_ask = self.ask_price
        else:
            if self.adjust_mid_spread:
                # Extract bid and ask price adjustments from the action
                mid_adjustment, spread_adjustment,_ = action
                mid_adjustment -= (self.n_price_adjustments-1)/2  # Shift the range to -k to +k
                spread_adjustment -= 1.0
                self.mid += mid_adjustment
                self.spread += spread_adjustment
                # print(self.mid)
                if self.fixed_spread:
                    self.spread = 6
                else:
                    self.spread = min(100,max(2,self.spread))
                # Update the bid and ask prices based on the adjustments
                self.bid_price = self.mid - self.spread/2
                self.ask_price = self.mid + self.spread/2
            else:
                # Extract bid and ask price adjustments from the action
                bid_adjustment, ask_adjustment,_ = action
                bid_adjustment -= (self.n_price_adjustments-1)/2  # Shift the range to -k to +k
                ask_adjustment -= (self.n_price_adjustments-1)/2  # Shift the range to -k to +k

                # Update the bid and ask prices based on the adjustments
                self.bid_price = self.bid_price + bid_adjustment
                self.ask_price = max(self.bid_price + 1, self.ask_price + bid_adjustment + ask_adjustment)
                
                # Penalize large price changes 
                # reward += -0.1*
        
        # Get noisy measurement of what the trader believes the price to be and behaves accordingly
        trader_price = self.get_noisy_measurement_of_price()
        skip_bayes = 1
        
        if trader_price > self.ask_price:
            trade_action = 1  # Buy
            if action == 0 and self.time_step > 0 and self.time_step%skip_bayes == 0:
                self.bayesian_belief_update("trader buy action")
        elif trader_price < self.bid_price:
            trade_action = -1  # Sell
            if action == 0 and self.time_step > 0 and self.time_step%skip_bayes == 0:
                self.bayesian_belief_update("trader sell action")
        else:
            trade_action = 0  # Hold
            if action == 0 and self.time_step > 0 and self.time_step%skip_bayes == 0:
                self.bayesian_belief_update("trader null action")
        
        
        # Update trade history and imbalances
        previous_imbalance = self.imbalance
        previous_short_term_imbalance = self.short_term_imbalance
        
        self.trade_history = np.roll(self.trade_history, shift=-1, axis=0)
        self.trade_history[-1] = trade_action
        if self.ema_base == -1:
            self.short_term_imbalance = np.sum(self.trade_history)
        else:
            self.short_term_imbalance = self.trade_history[-1] + self.ema_base * previous_short_term_imbalance
            self.short_term_imbalance = min(self.max_history_len,max(-self.max_history_len,round(self.short_term_imbalance[0])))
        self.imbalance += trade_action

        # Calculate monetary loss/profit
        prev_monetary_loss = self.cumulative_monetary_loss
        
        if trade_action == 1 and self.current_p_ext > self.ask_price:
            self.cumulative_monetary_loss -= (self.current_p_ext) - (self.ask_price)
        elif trade_action == -1 and self.current_p_ext < self.bid_price:
            self.cumulative_monetary_loss -= (self.bid_price) - (self.current_p_ext)
        elif trade_action == 1:
            self.cumulative_monetary_loss += (self.ask_price) - (self.current_p_ext)
        elif trade_action == -1:
            self.cumulative_monetary_loss += (self.current_p_ext) - (self.bid_price)
            
        monetary_loss_step = self.cumulative_monetary_loss - prev_monetary_loss 

        # Calculate the reward
        if self.use_endogynous:
            if self.use_short_term:
                reward -= np.square(self.short_term_imbalance) #- self.mu * ((self.ask_price - self.bid_price )) 
                #reward += np.square(previous_short_term_imbalance) - np.square(self.short_term_imbalance) #- self.mu * ((self.ask_price - self.bid_price )) 
            else:
                reward += np.square(previous_imbalance) - np.square(self.imbalance) #- self.mu * ((self.ask_price - self.bid_price ))        
        else:
            reward += monetary_loss_step #- self.mu * ((self.ask_price - self.bid_price ))
        
        if not self.fixed_spread:
            reward -= self.mu*(self.ask_price - self.bid_price)**self.spread_exp
            
        if not self.adjust_mid_spread:
            if trade_action == 1:
                reward += -0.1*self.ask_price
            elif trade_action == -1:
                reward += 0.1*self.bid_price
        
        if self.use_stored_path:
            if self.vary_informed:
                self.informed = self.informed_path[self.time_step]
            if self.vary_jump_prob:
                self.jump_prob = self.jump_prob_path[self.time_step]
        else:   
            if self.vary_informed:
                self.informed += np.random.choice([-0.01, 0.01])
                self.informed = np.clip(self.informed, 0.01, 0.99)
                self.informed_path[self.time_step]=(self.informed)
                
            if self.vary_jump_prob:
                self.jump_prob += np.random.choice([-0.01, 0.01])
                self.jump_prob = np.clip(self.jump_prob, 0.01, 0.99)
                self.jump_prob_arr[self.time_step]=(self.jump_prob)
            
        # Check if the episode is done
        done = self.time_step >= self.max_episode_len
        self.time_step += 1

        # Update the current true price
        if self.use_stored_path:
            next_p_ext = self.price_path[self.time_step]
        else:
            jump = np.random.choice(["jump up","jump down", "no jump"], p=[self.jump_prob/2, self.jump_prob/2, 1-self.jump_prob])
            next_p_ext = self.current_p_ext
            if jump == "jump up":
                next_p_ext = self.current_p_ext + self.jump_size#min(self.current_p_ext + 1,200)
            if jump == "jump down":
                next_p_ext = self.current_p_ext - self.jump_size#max(0,self.current_p_ext - 1)
        
        if action == 0:
            self.bayesian_belief_update("price jump")
            
        
        extra_dict["monetary_loss"] = monetary_loss_step
        extra_dict["ask"] = self.ask_price
        extra_dict["bid"] = self.bid_price
        extra_dict["spread"] = self.ask_price - self.bid_price
        extra_dict["mid"] = (self.ask_price + self.bid_price)/2
        extra_dict["p_ext"] = self.current_p_ext
        

        if not done:
            self.current_p_ext = next_p_ext
        
        if self.use_endogynous:
            if self.use_short_term:
                observation = (self.trade_history, self.short_term_imbalance)
            else:
                observation = (self.trade_history, self.imbalance)
        else:
            observation = (self.trade_history, self.short_term_imbalance)
        return observation, reward, done, extra_dict

def get_args():
    parser = argparse.ArgumentParser(description='Glosten-Milgrom market making simulation')

    parser.add_argument('--p_ext', type=float, default=100, help='Initial true price')
    parser.add_argument('--spread', type=float, default=2, help='Initial spread')
    parser.add_argument('--mu', type=float, default=0.1, help='Mu parameter')
    
    parser.add_argument('--spread_exp', type=float, default=2, help='Spread penalty exponent')
    parser.add_argument('--max_history_len', type=int, default=21, help='History length for calculating imbalance')
    parser.add_argument('--max_episode_len', type=int, default=2, help='Number of time slots')
    parser.add_argument('--max_episodes', type=int, default=120, help='Number of training episodes')
    parser.add_argument('--ema_base', type=int, default=-1, help='exponential moving average')
    
    parser.add_argument('--ALPHA', type=float, default=0.9, help='Percentage of informed traders')
    parser.add_argument('--vary_informed', type=bool, default=False, help='vary the informed trader proportion')
    
    parser.add_argument('--SIGMA', type=float, default=1.0, help='Probability of price jump')
    parser.add_argument('--vary_jump_prob', type=bool, default=False, help='vary the volatility')
    
    parser.add_argument('--jump_size', type=int, default=1, help='Size of price jump')
    parser.add_argument('--jump_at', type=int, default=-1, help='= -1 if no jumps, if positive, then jumps at that time by 1000*jump_size and stays constant')
    
    parser.add_argument('--use_short_term', type=bool, default=False, help='Use short-term imbalance')
    parser.add_argument('--use_endogynous', type=bool, default=False, help='Use endogenous variables')
    parser.add_argument('--n_price_adjustments', type=int, default=3, help='Number of actions to adjust mid price')
    parser.add_argument('--adjust_mid_spread', type=bool, default=False, help='Adjust mid + spread')
    parser.add_argument('--fixed_spread', type=bool, default=False, help='Fix the spread')
    
    parser.add_argument('--use_stored_path', type=bool, default=False, help='Use a generated sample path again')
    
    parser.add_argument('--compare', type=bool, default=False, help='Compare with !use_endogynous')
    parser.add_argument('--compare_with_bayes', type=bool, default=False, help='Compare with bayesian agent')
    
    parser.add_argument('--state_is_vec', type=bool, default=False, help='is state a vector')
    
    parser.add_argument('--special_string', type=str, help='Special string for output folder')
    
    parser.add_argument('--model_transfer', type=bool, default=False, help='Reuse the same agent')
    
    parser.add_argument('--agent_type', type=str, default="QT", help='RL agent type (QT, DQN, SARSA)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount rate of future rewards')
    parser.add_argument('--epsilon', type=float, default=0.9999, help='Starting exploration probability')
    
    parser.add_argument('--mode', type=str, default="valid", help='Mode for moving average calculation')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='checkpoint model after training iterations')
    
    parser.add_argument('--noise_type', type=str, default="Gaussian", help='Type of noise in trader price belief')
    # CHOCES FOR NOISE TYPE : "Bernoulli", "Gaussian", "Laplacian", "GeomGaussian"
    parser.add_argument('--noise_mean', type=float, default=0.0, help='mean of the noise')
    parser.add_argument('--noise_variance', type=float, default=25, help='variance of the noise')

    args = parser.parse_args()
    return args

