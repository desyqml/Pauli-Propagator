import matplotlib.pyplot as plt
import numpy as np


def plot_obs(prop):
    # Define bin edges
    bins = np.arange(max(prop.hist) + 2)

    # Plot histogram with aligned bars
    plt.hist(prop.hist, bins=bins, align="mid", rwidth=0.7, color = "gray", alpha=.6)
    plt.hist(prop.trim_hist, bins=bins, align="mid", rwidth=0.7, color = "royalblue", edgecolor="black", label="Trimmed")

    # Set ticks at the centers of bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, labels=[str(int(c)) for c in bin_centers])

    plt.xlabel("Pauli weight")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()