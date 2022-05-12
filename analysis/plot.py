
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rc('text', usetex=True)

def plot_width():

    def make_plot(ax, df, nlayers):
        df = df[df['model_layers'] == nlayers]
        
        vals = []
        widths = [2, 4, 8, 32, 64]
        sizes = []
        for width in widths:
            sizes.append(df[df["width"] == width]["epoch_4"].to_numpy().shape[0])
            vals.append(df[df["width"] == width]["epoch_4"].to_numpy())
    
        ax.boxplot(vals, positions=widths, widths=4.0)
        return min(sizes)
    
    df = pd.read_pickle("df.pkl")
    df = df[df["dataset_type"] == "generate_trellis"]
    
    df = df[df["alpha_num"] == 64.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharex=True, sharey=True)
    
    for (i, nlayers) in enumerate([2, 4, 8]):
        min_size = make_plot(axes[i], df, nlayers)
        axes[i].set_title(f"{nlayers} Layer Model (n={min_size})")
    
    axes[0].set_ylim([0.5, 1.0])
    axes[1].set_xlabel("Width")
    axes[0].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("plots/width.png")

def plot_alpha():

    def make_plot(ax, df, nlayers):
        df = df[df['model_layers'] == nlayers]
        
        vals = []
        alpha_nums = [2, 4, 8, 32, 64]
        sizes = []
        for alpha_num in alpha_nums:
            sizes.append(df[df["alpha_num"] == alpha_num]["epoch_4"].to_numpy().shape[0])
            vals.append(df[df["alpha_num"] == alpha_num]["epoch_4"].to_numpy())
    
        ax.boxplot(vals, positions=alpha_nums, widths=4.0)
        return min(sizes)
    
    df = pd.read_pickle("df.pkl")
    df = df[df["dataset_type"] == "generate_trellis"]
    df = df[df["width"] == 2.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharex=True, sharey=True)
    
    for (i, nlayers) in enumerate([2, 4, 8]):
        min_size = make_plot(axes[i], df, nlayers)
        axes[i].set_title(f"{nlayers} Layer Model (n={min_size})")
    
    #fig.suptitle("Transfomer on Trellis Performance Versus Alphabet Size")
    
    axes[1].set_xlabel("Alphabet Size")
    axes[2].set_ylabel("Accuracy")
    
    axes[0].set_ylim([0.5, 1.0])
    axes[1].set_xlabel("Alphabet Size")
    axes[0].set_ylabel("Accuracy")

    plt.tight_layout()

    plt.savefig("plots/alpha.png")

def plot_random():
    def make_plot(ax, df, nlayers):
        df = df[df['model_layers'] == nlayers]
        
        vals = []
        states = [4, 6, 8, 10]
        sizes = []
        for ns in states:
            sizes.append(df[df["n_states"] == ns]["epoch_4"].to_numpy().shape[0])
            vals.append(df[df["n_states"] == ns]["epoch_4"].to_numpy())
    
        ax.boxplot(vals, positions=states, widths=1.0)
        return min(sizes)
    
    df = pd.read_pickle("df.pkl")
    df = df[df["dataset_type"] != "generate_trellis"]

    se_df = df[df["model_name"] == "SimpleEncoder"]
    
    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharex=True, sharey=True)
    
    for (i, nlayers) in enumerate([2, 4, 8]):
        min_size = make_plot(axes[i], df, nlayers)
        axes[i].set_title(f"{nlayers} Layer Model (n={min_size})")
    
    axes[0].set_xlim([3.0, 11.0])
    axes[0].set_ylim([0.5, 1.0])
    axes[1].set_xlabel("Number of States")
    axes[0].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("plots/random.png")

def plot_random_bonus():

    def make_plot(ax, df):
        vals = []
        states = [4, 6, 8, 10]
        sizes = []
        for ns in states:
            sizes.append(df[df["n_states"] == ns]["epoch_4"].to_numpy().shape[0])
            vals.append(df[df["n_states"] == ns]["epoch_4"].to_numpy())
    
        ax.boxplot(vals, positions=states, widths=1.0)
        return min(sizes)
    
    df = pd.read_pickle("df.pkl")
    df = df[df["dataset_type"] != "generate_trellis"]

    lstm_df = df[df["model_name"] == "LSTM"]
    bert_df = df[df["model_name"] == "BERT"]
    
    fig, axes = plt.subplots(1, 2, figsize=(4, 3), sharex=True, sharey=True)
    
    min_size = make_plot(axes[0], lstm_df)
    axes[0].set_title(f"LSTM (n={min_size})")

    min_size = make_plot(axes[1], bert_df)
    axes[1].set_title(f"BERT (n={min_size})")
    
    axes[0].set_xlim([3.0, 11.0])
    axes[0].set_ylim([0.5, 1.0])
    axes[1].set_xlabel("Number of States")
    axes[0].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("plots/random_bonus.png")

def plot_lines():

    def make_plot_alpha(ax, df, nlayers, color):
        df = df[df["dataset_type"] == "generate_trellis"]
        df = df[df['model_layers'] == nlayers]
        
        vals = []
        alpha_nums = [2, 4, 8, 32, 64]
        sizes = []
        for alpha_num in alpha_nums:
            vals.append(np.mean(df[df["alpha_num"] == alpha_num]["epoch_4"].to_numpy()))

        ax.plot(alpha_nums, vals, color=color)
        ax.scatter(alpha_nums, vals, color=color, marker='x')

    def make_plot_alpha_lstm(ax, df, color):
        df = df[df["dataset_type"] == "generate_trellis"]
        df = df[df["model_name"] == "LSTM"]
        
        vals = []
        alpha_nums = [2, 4, 8, 32, 64]
        sizes = []
        for alpha_num in alpha_nums:
            vals.append(np.mean(df[df["alpha_num"] == alpha_num]["epoch_4"].to_numpy()))

        ax.plot(alpha_nums, vals, color=color)
        ax.scatter(alpha_nums, vals, color=color, marker='x')

    def make_plot_width(ax, df, nlayers, color):
        df = df[df["dataset_type"] == "generate_trellis"]
        df = df[df['model_layers'] == nlayers]
        df = df[df["alpha_num"] == 64.0]
        
        vals = []
        widths = [2, 4, 8, 32, 64]
        sizes = []
        for width in widths:
            vals.append(np.mean(df[df["width"] == width]["epoch_4"].to_numpy()))

        ax.plot(widths, vals, color=color)
        ax.scatter(widths, vals, color=color, marker='x')
        
    def make_plot_width_lstm(ax, df, color):
        df = df[df["dataset_type"] == "generate_trellis"]
        df = df[df["model_name"] == "LSTM"]
        df = df[df["alpha_num"] == 64.0]
        
        vals = []
        widths = [2, 4, 8, 32, 64]
        sizes = []
        for width in widths:
            vals.append(np.mean(df[df["width"] == width]["epoch_4"].to_numpy()))

        ax.plot(widths, vals, color=color)
        ax.scatter(widths, vals, color=color, marker='x')

    def make_plot_random(ax, df, nlayers, color):
        df = df[df["dataset_type"] != "generate_trellis"]
        df = df[df['model_layers'] == nlayers]
        
        vals = []
        states = [4, 6, 8, 10]
        sizes = []
        for ns in states:
            vals.append(np.mean(df[df["n_states"] == ns]["epoch_4"].to_numpy()))
    
        ax.plot(states, vals, color=color)
        ax.scatter(states, vals, color=color, marker='x')

    def make_plot_random_lstm(ax, df, color):
        df = df[df["dataset_type"] != "generate_trellis"]
        df = df[df["model_name"] == "LSTM"]
        
        vals = []
        states = [4, 6, 8, 10]
        sizes = []
        for ns in states:
            vals.append(np.mean(df[df["n_states"] == ns]["epoch_4"].to_numpy()))
    
        ax.plot(states, vals, color=color)
        ax.scatter(states, vals, color=color, marker='x')
    
    df = pd.read_pickle("df.pkl")
    
    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharex=False, sharey=True)

    colors = ['red', 'green', 'blue']
    for (i, nlayers) in enumerate([2, 4, 8]):
        make_plot_width(axes[0], df, nlayers, colors[i])
        make_plot_alpha(axes[1], df, nlayers, colors[i])
        make_plot_random(axes[2], df, nlayers, colors[i])

    make_plot_alpha_lstm(axes[1], df, 'black')
    make_plot_width_lstm(axes[0], df, 'black')
    make_plot_random_lstm(axes[2], df, 'black')
    
    axes[0].set_ylim([0.5, 1.0])
    axes[1].set_ylim([0.5, 1.0])
    axes[2].set_ylim([0.5, 1.0])
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Accuracy")

    axes[1].set_xlabel("Alphabet Size")
    axes[2].set_xlabel("Number of States")

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='black', lw=4),
                    ]
    
    axes[0].legend(custom_lines, ["2 Layer", "4 Layer", "8 Layer", "LSTM"])

    plt.tight_layout()
    plt.savefig("plots/trends.png")


def plot_alpha_lstm():

    def make_plot_alpha(ax, mdf):

        df = mdf[mdf["width"] == 2.0]
        alpha_nums = [2, 4, 8, 32, 64]
        sizes = []
        vals =[]
        for alpha_num in alpha_nums:
            sizes.append(df[df["alpha_num"] == alpha_num]["epoch_4"].to_numpy().shape[0])
            vals.append(df[df["alpha_num"] == alpha_num]["epoch_4"].to_numpy())
    
        ax.boxplot(vals, positions=alpha_nums, widths=4.0)
        return min(sizes)

    def make_plot_width(ax, mdf):

        df = mdf[mdf["alpha_num"] == 64.0]
        vals = []
        widths = [2, 4, 8, 32, 64]
        sizes = []
        for width in widths:
            sizes.append(df[df["width"] == width]["epoch_4"].to_numpy().shape[0])
            vals.append(df[df["width"] == width]["epoch_4"].to_numpy())
    
        ax.boxplot(vals, positions=widths, widths=4.0)
        return min(sizes)
    
    df = pd.read_pickle("df.pkl")
    df = df[df["dataset_type"] == "generate_trellis"]
    df = df[df["model_name"] == "LSTM"]
    
    fig, axes = plt.subplots(1, 2, figsize=(4, 3), sharex=True, sharey=True)
    
    min_size = make_plot_alpha(axes[0], df)
    axes[0].set_title(f"LSTM (n={min_size})")
    min_size = make_plot_width(axes[1], df)
    axes[1].set_title(f"LSTM (n={min_size})")
    
    #fig.suptitle("Transfomer on Trellis Performance Versus Alphabet Size")
    
    #axes[1].set_xlabel("Alphabet Size")
    #axes[2].set_ylabel("Accuracy")
    
    axes[0].set_ylim([0.5, 1.0])
    axes[0].set_xlabel("Alphabet Size")
    axes[1].set_xlabel("Width")
    axes[0].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("plots/lstm_scaling.png")

if __name__ == '__main__':
    plot_width()
    plt.cla()
    plot_alpha()
    plt.cla()
    plot_random()
    plt.cla()
    plot_random_bonus()
    plt.cla()
    plot_lines()
    plt.cla()
    plot_alpha_lstm()

