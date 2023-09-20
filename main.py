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
import sys,csv,math
import numpy as np
import os

from tqdm import tqdm

from env import GlostenMilgromEnv
from agent import DQN_Agent,QLearningAgent,QLearningAgentUpperConf,BayesianAgent, save_checkpoint, load_checkpoint

def get_args():
    parser = argparse.ArgumentParser(description='Glosten-Milgrom market making simulation')

    parser.add_argument('--p_ext', type=float, default=100, help='Initial true price')
    parser.add_argument('--spread', type=float, default=2, help='Initial spread')
    parser.add_argument('--mu', type=float, default=0.1, help='Mu parameter')
    
    parser.add_argument('--spread_exp', type=float, default=2, help='Spread penalty exponent')
    parser.add_argument('--max_history_len', type=int, default=20, help='History length for calculating imbalance')
    parser.add_argument('--max_episode_len', type=int, default=200000, help='Number of time slots')
    parser.add_argument('--max_episodes', type=int, default=1, help='Number of training episodes')
    parser.add_argument('--ema_base', type=int, default=-1, help='exponential moving average')
    
    parser.add_argument('--informed', type=float, default=0.9, help='Percentage of informed traders')
    parser.add_argument('--vary_informed', type=bool, default=False, help='vary the informed trader proportion')
    
    parser.add_argument('--jump_prob', type=float, default=1.0, help='Probability of price jump')
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
    parser.add_argument('--noise_mean', type=float, default="valid", help='mean of the noise')
    parser.add_argument('--noise_variance', type=float, default=1000, help='variance of the noise')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    moving_avg = int(args.max_episode_len/200)
    if moving_avg < 10:
        moving_avg = 1
    mode="valid"
    # Create the environment
    env = GlostenMilgromEnv(
        args.p_ext, 
        args.spread, 
        args.mu, 
        jump_prob=args.jump_prob, 
        informed=args.informed, 
        max_episode_len=args.max_episode_len, 
        max_history_len=args.max_history_len,
        use_short_term=args.use_short_term,
        use_endogynous=args.use_endogynous,
        n_price_adjustments=args.n_price_adjustments,
        adjust_mid_spread=args.adjust_mid_spread,
        fixed_spread=args.fixed_spread,
        use_stored_path=args.use_stored_path,
        spread_exp=args.spread_exp,
        jump_size=args.jump_size,
        vary_informed=args.vary_informed,
        vary_jump_prob=args.vary_jump_prob,
        ema_base=args.ema_base,
        compare_with_bayes = args.compare_with_bayes,
        jump_at=args.jump_at
    )

    figure_path = "modelFreeGM/informed_{0}_jump_{1}_mu_{2}/fixedSpread_{10}_useShortTerm_{3}_useEndo_{4}_maxHistoryLen_{5}/agentType_{6}_alpha_{7}_gamma_{8}_epsilon_{9}".format(
        args.informed,
        args.jump_prob,
        args.mu,
        args.use_short_term,
        args.use_endogynous,
        args.max_history_len,
        args.agent_type,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.fixed_spread
    )

    if args.special_string is not None:
        figure_path = figure_path + "/{0}".format(args.special_string)

    if args.ema_base != -1:
        figure_path = figure_path + "/ema_base_{0}".format(args.ema_base)
        
    if not args.adjust_mid_spread:
        figure_path = figure_path + "/direct_ask_bid_control"

    os.makedirs(figure_path , exist_ok=True)
    # Create the agent


    if args.model_transfer:
        agent = load_checkpoint(figure_path+"/model_ckpt.pkl")
    else:
        if args.agent_type == "QT": # tabular q learning with epsilon exploration
            n_states = 2*args.max_history_len + 1  # Define the number of discrete states for the given history window
            agent = QLearningAgent(
                n_actions=[env.action_space[0].n,env.action_space[1].n], 
                n_states=n_states, 
                alpha=args.alpha, 
                gamma=args.gamma, 
                epsilon=args.epsilon
            )
            comparison_agent = QLearningAgent(
                n_actions=[env.action_space[0].n,env.action_space[1].n], 
                n_states=n_states, 
                alpha=args.alpha, 
                gamma=args.gamma, 
                epsilon=args.epsilon
            )
        elif args.agent_type == "QUCB": # tabular q learning + ucb exploration
            n_states = 2*args.max_history_len + 1  # Define the number of discrete states for the given history window
            agent = QLearningAgentUpperConf(
                n_actions=[env.action_space[0].n,env.action_space[1].n], 
                n_states=n_states, 
                alpha=args.alpha, 
                gamma=args.gamma, 
                c=c
            )
            comparison_agent = QLearningAgentUpperConf(
                n_actions=[env.action_space[0].n,env.action_space[1].n], 
                n_states=n_states, 
                alpha=args.alpha, 
                gamma=args.gamma, 
                c=c
            )
        elif args.agent_type == "DQN":
            state_dim = 1
            state_is_vec = True
            agent = DQN_Agent(
                args.max_history_len,
                args.n_price_adjustments,
                num_adjustments=args.n_price_adjustments,
                window=args.max_history_len,
                hidden_size=64,
                lr=1e-3,
                gamma=args.gamma,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995
            )
        elif args.agent_type == "SARSA":
            state_dim = 1
            agent = SARSA_agent(
                n_actions=[env.action_space[0].n,env.action_space[1].n],
                state_dim=state_dim,
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon=args.eargs.psilon
            )
        elif args.agent_type == "UCRL":
            pass
        elif args.agent_type == "TD":
            pass
        elif args.agent_type == "AI":
            pass
        elif args.agent_type == "BA":
            n_states = 2*args.max_history_len + 1
            agent = BayesianAgent(
                n_actions=[env.action_space[0].n,env.action_space[1].n], 
                n_states=n_states, 
                alpha=args.alpha, 
                gamma=args.gamma, 
                epsilon=args.epsilon
            )
        else:
            print("ERROR_UNKNOWN_AGENT_TYPE")
        if args.compare_with_bayes:
            n_states_bayes = 2*args.max_history_len + 1
            bayesian_agent = BayesianAgent(
                n_actions=[env.action_space[0].n,env.action_space[1].n], 
                n_states=n_states_bayes, 
                alpha=args.alpha, 
                gamma=args.gamma, 
                epsilon=args.epsilon
            )



    # Train the agent for some number of episodes - ideally should only need one episode for training the network
    # output detailed plots for the last episode

    total_rewards = []
    monetary_losses = []

    total_rewards_compare = []
    monetary_losses_compare = []

    total_rewards_bayes = []
    monetary_losses_bayes = []

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

    def make_figure_and_save_model():

        save_path = figure_path + "/model_ckpt.pkl"

        # Calculate the average total reward and monetary loss
        average_total_reward = np.mean(total_rewards)
        average_monetary_loss = np.mean(monetary_losses)
        print("Average total reward:", average_total_reward/max_episode_len)
        print("Average monetary loss:", average_monetary_loss/max_episode_len)
        print("Mean spread",np.mean(np.array(spread_vs_time)))
        print("Mean mid dev",np.mean(abs(np.array(mid_price_vs_time)-np.array(p_ext_vs_time))))

        if args.compare:
            average_total_reward_compare = np.mean(total_rewards_compare)
            average_monetary_loss_compare = np.mean(monetary_losses_compare)
            print("Average total reward reference:", average_total_reward_compare/max_episode_len)
            print("Average monetary loss reference:", average_monetary_loss_compare/max_episode_len)
        if args.compare_with_bayes:
            average_total_reward_bayes = np.mean(total_rewards_bayes)
            average_monetary_loss_bayes = np.mean(monetary_losses_bayes)
            print("Average total reward bayes:", average_total_reward_bayes/max_episode_len)
            print("Average monetary loss bayes:", average_monetary_loss_bayes/max_episode_len)

        # Plot the average monetary loss over all episodes
        plt.plot(monetary_losses)
        if args.compare:
            plt.plot(monetary_losses_compare)
        if args.compare_with_bayes:
            plt.plot(monetary_losses_bayes)
        plt.xlabel("Episode Number")
        plt.ylabel("Average Monetary Loss vs time")
        plt.title("Average Monetary Loss over Episodes")
        plt.legend()
        filename = "Monetary_Loss.pdf"
        file_path = os.path.join(figure_path, filename)
        plt.savefig(file_path)  
        plt.show()
         

        # Plot the average total reward over all episodes
        plt.plot(total_rewards)
        if args.compare:
            plt.plot(total_rewards_compare)
        if args.compare_with_bayes:
            plt.plot(total_rewards_bayes)
        plt.xlabel("Episode Number")
        plt.ylabel("Average total reward")
        plt.title("Average total reward over Episodes")
        plt.legend()
        filename = "Total_Reward.pdf"
        file_path = os.path.join(figure_path, filename)
        plt.savefig(file_path)
        plt.show()



        # Plot the monetary loss over time for the last episode
        plt.plot(np.convolve(np.array(monetary_losses_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="RL")
        # if args.compare:
        #     plt.plot(np.convolve(np.array(monetary_losses_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")
        if args.compare_with_bayes:
            plt.plot(np.convolve(np.array(monetary_losses_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode),label="Bayes")
        plt.xlabel("Time")
        plt.ylabel("Monetary Loss")
        plt.title("Monetary Loss over time")
        plt.ylim((-5,5))
        plt.legend()
        filename = "Loss_Vs_time.pdf"
        file_path = os.path.join(figure_path, filename)
        plt.savefig(file_path)
        plt.show()


        # Plot the reward over time for the last episode
        plt.plot(np.convolve(np.array(rewards_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="orig")
        if args.compare:
            plt.plot(np.convolve(np.array(rewards_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")
        plt.xlabel("Time")
        plt.ylabel("Reward")#
        plt.title("Reward over time")
        plt.legend()
        filename = "Reward_Vs_time.pdf"
        file_path = os.path.join(figure_path, filename)
        plt.savefig(file_path)
        plt.show()


        # Plot the spread over time for the last episode
        plt.plot(np.convolve(np.array(spread_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="RL")# spread should decay with time
        # if args.compare:
        #     plt.plot(np.convolve(np.array(spread_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")# spread should decay with time
        if args.compare_with_bayes:
            plt.plot(np.convolve(np.array(spread_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode),label="Bayes")# spread should decay with time
        plt.xlabel("Time")
        plt.ylabel("Spread")
        plt.title("Spread over time : one sample path")
        plt.legend()
        filename = "Spread_Vs_time.pdf"
        file_path = os.path.join(figure_path, filename)
        plt.savefig(file_path)
        plt.show()

        # Plot the ask,bid and external price over time for the last episode
        plt.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="P_ext")
        plt.plot(np.convolve(np.array(ask_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="Ask")
        plt.plot(np.convolve(np.array(bid_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="Bid")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("RL Ask and Bid over time : one sample path")
        plt.legend()
        filename = "AskBid_Vs_time.pdf"
        file_path = os.path.join(figure_path, filename)
        plt.savefig(file_path)
        plt.show()
        if args.compare:
            plt.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="P_ext")
            plt.plot(np.convolve(np.array(ask_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode), label="Ask")
            plt.plot(np.convolve(np.array(bid_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode), label="Bid")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.title("RL Ask and Bid over time : one sample path (reward has p_ext)")
            plt.legend()
            filename = "AskBid_Vs_time_compare.pdf"
            file_path = os.path.join(figure_path, filename)
            plt.savefig(file_path)
            plt.show()
        if args.compare_with_bayes:
            plt.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="P_ext")
            plt.plot(np.convolve(np.array(ask_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode), label="Ask")
            plt.plot(np.convolve(np.array(bid_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode), label="Bid")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.title("Bayes Ask and Bid over time : one sample path")
            plt.legend()
            filename = "AskBid_Vs_time_bayes.pdf"
            file_path = os.path.join(figure_path, filename)
            plt.savefig(file_path)
            plt.show()

        # Plot the mid and ext price over time for the last episode
        plt.plot(abs(np.convolve((np.array(mid_price_vs_time)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode)), label="RL")# need to make sure E[(mid-p_ext)^2]-> 0 as t becomes large for the last episode
        #plt.semilogy(np.convolve(abs(np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode), label="compare")# need to make sure E[(mid-p_ext)^2]-> 0 as t becomes large for the last episode
        if args.compare_with_bayes:
            plt.plot(abs(np.convolve((np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode)), label="Bayes")# need to make sure E[(mid-p_ext)^2]-> 0 as t becomes large for the last episode

        #plt.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="P_ext")# this should be as near to mid as possible
        plt.xlabel("Time")
        plt.ylabel("mid price - ext price")
        #plt.yscale("log")
        plt.title("Mid price deviation over time")
        plt.legend()
        filename = "Mid_Vs_time.pdf"
        file_path = os.path.join(figure_path, filename)
        plt.savefig(file_path)
        plt.show()

        if env.vary_jump_prob:
            plt.plot(np.convolve(np.array(env.jump_prob_arr),np.ones(moving_avg)/moving_avg,mode=mode), label="Volatility")
        if env.vary_informed:
            plt.plot(np.convolve(np.array(env.informed_arr),np.ones(moving_avg)/moving_avg,mode=mode), label="Informedness")

        if env.vary_informed or env.vary_jump_prob:
            plt.xlabel("Time")
            plt.ylabel("Parameter Value")
            #plt.yscale("log")
            plt.title("Variable volatility and informedness")
            plt.legend()
            filename = "Params_Vs_time.pdf"
            file_path = os.path.join(figure_path, filename)
            plt.savefig(file_path)
            plt.show()


        save_checkpoint(agent,save_path)
        #____________
        # spread_mean=[]
        # mid_dev_mean=[]
        # alphas=[]
        # sigmas=[]

        # spread_mean.append(np.mean(np.array(spread_vs_time)))
        # mid_dev_mean.append(np.mean(abs(np.array(mid_price_vs_time))))
        # alphas.append(informed)
        # sigmas.append(jump_prob)
        # spread_mean



    if not state_is_vec:
        n_states = agent.n_states
    for episode in tqdm(range(args.max_episodes)):
        env.reset()
        env.resetAllVars()
        
        if not state_is_vec:
            state = normalize_trade_imbalance(env.imbalance, n_states)
        else:
            state = torch.zeros(1,args.max_history_len)
        done = False
        total_reward = 0
        
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
                    
            if episode == args.max_episodes-1:
                p_ext_vs_time.append(extra_dict["p_ext"])
                rewards_vs_time.append(reward)
                monetary_losses_vs_time.append(extra_dict["monetary_loss"])
                spread_vs_time.append(extra_dict["spread"])
                ask_vs_time.append(extra_dict["ask"])
                bid_vs_time.append(extra_dict["bid"])
                mid_price_vs_time.append(extra_dict["mid"])
                # print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
                
        total_rewards.append(total_reward)
        monetary_losses.append(env.cumulative_monetary_loss)

    if args.compare :
        env.use_endogynous = not env.use_endogynous
        
        n_states = comparison_agent.n_states
        for episode in range(args.max_episodes):
            env.reset()
            env.resetAllVars()

            state = normalize_trade_imbalance(env.imbalance, n_states)
            done = False
            total_reward = 0

            time = 0

            while not done:
                action = comparison_agent.choose_action(state,epsilon=epsilon**time)

                (next_trade_history, next_imbalance), reward, done, extra_dict = env.step(action)
                next_state = normalize_trade_imbalance(next_imbalance, n_states)

                comparison_agent.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                time += 1

                if episode == args.max_episodes-1:
                    rewards_vs_time_compare.append(reward)
                    monetary_losses_vs_time_compare.append(extra_dict["monetary_loss"])
                    spread_vs_time_compare.append(extra_dict["spread"])
                    ask_vs_time_compare.append(extra_dict["ask"])
                    bid_vs_time_compare.append(extra_dict["bid"])
                    mid_price_vs_time_compare.append(extra_dict["mid"])


            total_rewards_compare.append(total_reward)
            monetary_losses_compare.append(env.cumulative_monetary_loss)

    if args.compare_with_bayes :
        env.use_endogynous = not env.use_endogynous
        
        n_states = bayesian_agent.n_states
        for episode in range(args.max_episodes):
            env.reset()
            env.resetAllVars()

            state = normalize_trade_imbalance(env.imbalance, n_states)
            done = False
            total_reward = 0

            time = 0

            while not done:
                action = bayesian_agent.choose_action(state,epsilon=epsilon**time)

                (next_trade_history, next_imbalance), reward, done, extra_dict = env.step(action)
                next_state = normalize_trade_imbalance(next_imbalance, n_states)

                bayesian_agent.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                time += 1

                if episode == args.max_episodes-1:
                    rewards_vs_time_bayes.append(reward)
                    monetary_losses_vs_time_bayes.append(extra_dict["monetary_loss"])
                    spread_vs_time_bayes.append(extra_dict["spread"])
                    ask_vs_time_bayes.append(extra_dict["ask"])
                    bid_vs_time_bayes.append(extra_dict["bid"])
                    mid_price_vs_time_bayes.append(extra_dict["mid"])


            total_rewards_bayes.append(total_reward)
            monetary_losses_bayes.append(env.cumulative_monetary_loss)



