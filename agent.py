import gym
import numpy as np
from gym.spaces import Discrete, Tuple, MultiDiscrete
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pickle

import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys,csv,math
import numpy as np
import os

import random

def save_checkpoint(agent, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(agent, f)

def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    return agent

class CFMM:
    """docstring for CFMM"""
    def __init__(self, reserves, type, params, fees):
        super(CFMM, self).__init__()
        self.reserves = reserves
        self.type = "mean"# mean, sum
        self.fees = fees
        self.params = params

    def update(self,trader_price,trade_size=-1):
        if self.type == "mean":
            fees = self.fees
            theta = self.params[0]
            x_0 = self.reserves[0]
            y_0 = self.reserves[1]
            k = (x_0**theta) * (y_0**(1-theta))
            current_price = (theta*x_0)/((1-theta)*y_0)

            if trader_price > current_price/(1-fees):
                x_1 = k*(theta/(trader_price*(1-theta)/(1-fees)))**(1-theta)
                y_1 = k*((trader_price*(1-theta)/(1-fees))/theta)**(theta)
                x_1 = x_0 + (x_1-x_0)/(1-fees)

                self.reserves[0] = x_1
                self.reserves[1] = y_1
            elif trader_price < current_price*(1-fees):
                x_1 = k*(theta/(trader_price*(1-theta)*(1-fees)))**(1-theta)
                y_1 = k*(trader_price*(1-theta)*(1-fees)/theta)**(theta)
                y_1 = y_0 + (y_1-y_0)/(1-fees)

                self.reserves[0] = x_1
                self.reserves[1] = y_1
            else:
                x_1 = x_0
                y_1 = y_0

            return (x_1-x_0,y_1-y_0)

    def modify(self,params):
        self.params = params
        


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DQN_Agent:
    def __init__(
        self,
        state_size,
        action_size,
        num_adjustments=1,
        window=20,
        hidden_size=64,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9999
    ):
        action_size = (num_adjustments)**2
        self.state_size = state_size
        self.action_size = action_size
        self.window = window
        self.num_adjustments = num_adjustments
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_network = DQN( window, action_size, hidden_size)
        self.target_network = DQN( window, action_size, hidden_size)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < self.epsilon:
            flat_action = np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.q_network(state)
            flat_action = torch.argmax(action_values).item()
        return int(flat_action/(self.num_adjustments)),flat_action%(self.num_adjustments),-1
        
    
    def update(self, state, action, reward, next_state, done=False):
        d = 0
        if done:
            d = 1
        action = self.num_adjustments*action[0]+action[1]
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward]).float()
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        done = torch.tensor([done], dtype=torch.bool)
        
        q_values = self.q_network(state).squeeze(0)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_state).squeeze(0)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + self.gamma * next_q_value * (1 - d)
        
        loss = self.loss_fn(q_value, target_q_value).float()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network
        self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

        

class PPO_Agent(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        num_adjustments=1,
        window=20,
        hidden_size=64,
        lr=1e-3,
        eps_clip=0.2,
        gamma=0.99
    ):
        super(PPO_Agent, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.window = window
        self.num_adjustments = num_adjustments
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        self.actor = nn.Sequential(
            nn.Linear(state_size * window, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_size * window, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor(action))
        entropy = dist.entropy()
        value = self.critic(state)
        return log_prob, entropy, value.squeeze(1)

    def update(self, memory):
        states, actions, rewards, log_probs_old, dones = memory.sample()
        
        # Calculate discounted rewards
        G = []
        g = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + self.gamma * g * (1 - d)
            G.insert(0, g)
        G = torch.tensor(G)
        
        for _ in range(10):  # Number of PPO epochs
            log_probs, entropies, values = self.evaluate(states, actions)
            advantages = G - values.detach()
            
            # Calculate the ratio
            ratio = torch.exp(log_probs - torch.tensor(log_probs_old))
            
            # Calculate the surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate the actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, G)
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropies.mean()
            
            # Update the network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class QLearningAgent:
    def __init__(
        self,
        n_actions,
        n_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        variable_trade_size=False,
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.variable_trade_size = variable_trade_size
        self.q_table = self.init_q_table(n_states, n_actions)

    def init_q_table(self, n_states, n_actions):
        q_table = {}
        for state in range(n_states):
            q_table[state] = {}
            for action_0 in range(n_actions[0]):
                q_table[state][action_0] = {}
                for action_1 in range(n_actions[1]):
                    if self.variable_trade_size:
                        q_table[state][action_0][action_1] = {}
                        for action_2 in range(n_actions[2]):
                            q_table[state][action_0][action_1][action_2] = 0
                    else:
                        q_table[state][action_0][action_1] = 0
        return q_table

    def choose_action(self, state, epsilon=-1):
        
        if epsilon == -1:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            if self.variable_trade_size:
                return np.random.choice(self.n_actions[0]), np.random.choice(self.n_actions[1]), np.random.choice(self.n_actions[2])  # Explore with prob epsilon
            else:
                return np.random.choice(self.n_actions[0]), np.random.choice(self.n_actions[1]), -1  # Explore with prob epsilon
        else:
            action_values = self.q_table[state]

            # Extract all Q-values into a list
            q_values = []
            for action_0 in action_values:
                for action_1 in action_values[action_0]:
                    if self.variable_trade_size:
                        for action_2 in action_values[action_0][action_1]:
                            q_values.append((action_values[action_0][action_1][action_2], (action_0, action_1, action_2)))
                    else:
                        q_values.append((action_values[action_0][action_1], (action_0, action_1, -1)))

            # Get the max Q-value
            max_q_value = max(q_values, key=lambda item: item[0])[0]

            # Get all actions that have the max Q-value
            max_actions = [action for q_value, action in q_values if q_value == max_q_value]

            # Choose a random action from those with the max Q-value
            max_action = random.choice(max_actions)

            return max_action  # Exploit
            # action_values = self.q_table[state]
            
            # action0 = max(action_values, key=lambda x: max(action_values[x][a][b].values())) 
            # mv = max(action_values[action0].values())
            # action0 = random.choice([k for (k, v) in action_values.items() if max(v.values()) == mv])

            # if self.variable_trade_size:
            #     action1 = random.choice([k for (k, v) in action_values[action0].items() if max(v.values()) == mv])
            # else:
            #     action1 = random.choice([k for (k, v) in action_values[action0].items() if v == mv])

            # action2 = -1
            # if self.variable_trade_size:
            #     action2 = random.choice([k for (k, v) in action_values[action0][action1].items() if v == mv])

            # return action0,action1,action2  # Exploit

    def update(self, state, action, reward, next_state):
        if self.variable_trade_size:
            predict = self.q_table[state][action[0]][action[1]][action[2]]
            target = reward + self.gamma * max(self.q_table[next_state][a][b][c] for a in range(self.n_actions[0]) for b in range(self.n_actions[1]) for c in range(self.n_actions[2]) )
            self.q_table[state][action[0]][action[1]][action[2]] += self.alpha * (target - predict)
        else:
            predict = self.q_table[state][action[0]][action[1]]
            target = reward + self.gamma * max(self.q_table[next_state][a][b] for a in range(self.n_actions[0]) for b in range(self.n_actions[1]))
            self.q_table[state][action[0]][action[1]] += self.alpha * (target - predict)
        
        

class QLearningAgentUpperConf:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, c=0.1, epsilon=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.c = c
        self.Q = np.ones((n_states, n_actions[0],n_actions[1])) * 1/(1-gamma)
        self.Q_hat = np.ones((n_states, n_actions[0],n_actions[1])) * 1/(1-gamma)
        self.state_visit_count = np.zeros(n_states)
        self.state_action_count = np.zeros((n_states,n_actions[0],n_actions[1]))
        self.epsilon=epsilon

    def choose_action(self, state, epsilon = 0):
        if self.state_visit_count[state] == 0:
            self.state_visit_count[state] += 1
            return (np.random.randint(self.n_actions[0]),np.random.randint(self.n_actions[1]))
        
        UCB_values = self.Q_hat[state]
        action_candidates_flat = np.flatnonzero(UCB_values == UCB_values.max())
        action_idx = np.random.choice(action_candidates_flat)
        action = (int(action_idx/self.n_actions[1]),action_idx%self.n_actions[1])
        
        return action

    def update(self, state, action, reward, next_state):
        self.state_action_count[state][action] += 1
        
        exploration_term = self.c / (1-self.gamma) * np.sqrt(np.log(self.state_action_count[state][action]) / self.state_action_count[state][action])
        
        target = reward + exploration_term + self.gamma * np.max(self.Q_hat[next_state])
        error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * 1/(self.state_action_count[state][action]) * error
        self.Q_hat[state][action] = min(self.Q[state][action],self.Q_hat[state][action])
        
class BayesianAgent:
    def __init__(
        self,
        n_actions,
        n_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, epsilon=-1):
        return 0

    def update(self, state, action, reward, next_state):
        pass
        