import pandas as pd
import matplotlib.pyplot as plt

import os

def get_average_plots():

    # Create directory to save plots if it doesn't exist
    if not os.path.exists("average_plots"):
        os.mkdir("average_plots")

    # Lines to adjust font sizes (uncomment as required)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14

    MARKER_SIZE = 6  # Adjust this value as desired
    MARKER_STYLES = {
        "_rl": "o",
        "_compare": "s",
        "_bayes": "^"
    }


    # Specify the column names based on the new scheme
    cols = ['alpha', 'sigma', '<ignore1>', '<ignore2>', 'loss_rl', '<ignore3>', 'spread_rl', '<ignore4>', 'mid_dev_rl', 
           '<ignore5>', '<ignore6>', 'loss_compare', '<ignore7>', 'spread_compare', '<ignore8>', 'mid_dev_compare',
           '<ignore9>', '<ignore10>', 'loss_bayes', '<ignore11>', 'spread_bayes', '<ignore12>', 'mid_dev_bayes']

    # Read the CSV with specified columns and no header
    df = pd.read_csv('losses_and_spreads_modified.csv', names=cols, header=None)

    LOSS_MULTIPLIER = 0.1 #convert to percentage loss - (average price is 1000)

    # Adjusting the loss columns by the multiplier
    df['loss_rl'] *= LOSS_MULTIPLIER
    df['loss_compare'] *= LOSS_MULTIPLIER
    df['loss_bayes'] *= LOSS_MULTIPLIER


    # Drop the columns marked as '<ignore>'
    df = df.drop(columns=[col for col in df.columns if 'ignore' in col])

    # Drop duplicates, keeping only the last occurrence of alpha, sigma pair
    df = df.drop_duplicates(subset=['alpha', 'sigma'], keep='last')

    # List of metrics we're interested in
    metrics = ['loss', 'spread']

    # Legend labels mapping
    legend_map = {
        "_rl": "Ours (Q-learning)",
        "_compare": "Loss oracle (Q-learning)",
        "_bayes": "Ours (Bayesian)"
    }

    # Function to format and show plots
    def show_plot(xlabel, ylabel, title, metric):
        plt.axhline(0, color='gray', linewidth=0.8)  # Line at y=0
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if metric == 'loss':
            y_min, y_max = plt.ylim()
            y_max = max(abs(y_min), abs(y_max))
            plt.ylim(-y_max, y_max)
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: ["Ours (Bayesian)", "Ours (Q-learning)", "Loss oracle (Q-learning)"].index(t[0])))
        plt.legend(handles, labels)
        plt.tight_layout()

    # 1. Plot vs. sigma for each distinct alpha
    alphas = df['alpha'].unique()
    for alpha in alphas:
        df_alpha = df[df['alpha'] == alpha].sort_values(by='sigma')
        for idx, metric in enumerate(metrics, 1):
            plt.figure(figsize=(8, 5))
            for key, label in legend_map.items():
                plt.plot(df_alpha['sigma'], df_alpha[f'{metric}{key}'], label=label, marker=MARKER_STYLES[key], markersize=MARKER_SIZE)
            ylabel = "% Monetary Loss" if metric == "loss" else "Spread"
            show_plot("Volatility ($\sigma$)", ylabel, "", metric)  # Removed title from show_plot
            plt.savefig(f"average_plots/{metric}_vs_sigma_for_alpha_{alpha}.pdf")
            plt.close()

    # 2. Plot vs. sigma, averaged over alpha values in [0.4, 0.9]
    alpha_mask = (df['alpha'] >= 0.4) & (df['alpha'] <= 0.9)
    df_alpha_filtered = df[alpha_mask].groupby('sigma').mean().reset_index()
    for idx, metric in enumerate(metrics, 1):
        plt.figure(figsize=(8, 5))
        for key, label in legend_map.items():
            plt.plot(df_alpha_filtered['sigma'], df_alpha_filtered[f'{metric}{key}'], label=label, marker=MARKER_STYLES[key], markersize=MARKER_SIZE)
        ylabel = "% Monetary Loss" if metric == "loss" else "Spread"
        show_plot("Volatility ($\sigma$)", ylabel, "", metric)  # Removed title from show_plot
        plt.savefig(f"average_plots/{metric}_vs_sigma_avg_alpha.pdf")
        plt.close()

    # 3. Plot vs. alpha, averaged over sigma values in [0.1, 1.0]
    sigma_mask = (df['sigma'] >= 0.1) & (df['sigma'] <= 1.0)
    df_sigma_filtered = df[sigma_mask].groupby('alpha').mean().reset_index()
    for idx, metric in enumerate(metrics, 1):
        plt.figure(figsize=(8, 5))
        for key, label in legend_map.items():
            plt.plot(df_sigma_filtered['alpha'], df_sigma_filtered[f'{metric}{key}'], label=label, marker=MARKER_STYLES[key], markersize=MARKER_SIZE)
        ylabel = "Monetary Loss" if metric == "loss" else "Spread"
        show_plot("Informedness ($\\alpha$)", ylabel, "", metric)  # Removed title from show_plot
        plt.savefig(f"average_plots/{metric}_vs_alpha_avg_sigma.pdf")
        plt.close()

#-----#
get_average_plots()
