
import pandas as pd
import matplotlib.pyplot as plt

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
    
        ax.boxplot(vals, positions=widths, widths=2.0)
        return min(sizes)
    
    df = pd.read_pickle("df.pkl")
    df = df[df["dataset_type"] == "generate_trellis"]
    
    df = df[df["alpha_num"] == 64.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    
    for (i, nlayers) in enumerate([2, 4, 8]):
        min_size = make_plot(axes[i], df, nlayers)
        axes[i].set_title(f"{nlayers} Layer Model (n={min_size})")
    
    fig.suptitle("Transfomer on Trellis Performance Versus Width")
    
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
    
        ax.boxplot(vals, positions=alpha_nums, widths=2.0)
        return min(sizes)
    
    df = pd.read_pickle("df.pkl")
    df = df[df["dataset_type"] == "generate_trellis"]
    df = df[df["width"] == 2.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    
    for (i, nlayers) in enumerate([2, 4, 8]):
        min_size = make_plot(axes[i], df, nlayers)
        axes[i].set_title(f"{nlayers} Layer Model (n={min_size})")
    
    fig.suptitle("Transfomer on Trellis Performance Versus Alphabet Size")
    
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
    df = df[df["model_name"] == "SimpleEncoder"]
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    
    for (i, nlayers) in enumerate([2, 4, 8]):
        min_size = make_plot(axes[i], df, nlayers)
        axes[i].set_title(f"{nlayers} Layer Model (n={min_size})")
    
    plt.suptitle("Transfomer on Binary DFA Performance Versus Number of States")
    
    axes[0].set_xlim([3.0, 11.0])
    axes[0].set_ylim([0.5, 1.0])
    axes[1].set_xlabel("Number of States")
    axes[0].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("plots/random.png")

if __name__ == '__main__':
    plot_width()
    plt.cla()
    plot_alpha()
    plt.cla()
    plot_random()
