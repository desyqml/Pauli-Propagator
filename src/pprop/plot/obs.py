import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def plot_obsW(prop, untrim=True, log=False):
    # Define bin edges
    bins = np.arange(max(prop.histW) + 2)

    # Plot histogram with aligned bars
    if untrim:
        plt.hist(
            prop.histW,
            bins=bins,
            align="mid",
            rwidth=0.7,
            color="gray",
            alpha=0.6,
            log=log,
        )
    plt.hist(
        prop.trim_histW,
        bins=bins,
        align="mid",
        rwidth=0.7,
        color="royalblue",
        edgecolor="black",
        label="Trimmed",
        log=log,
    )

    # Set ticks at the centers of bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, labels=[str(int(c)) for c in bin_centers])

    plt.xlabel("Pauli weight")
    plt.ylabel("Counts (log scale)" if log else "Counts")
    plt.legend()
    plt.show()


def plot_obsF(prop, log=False):
    # Define bin edges
    bins = np.arange(max(prop.histF) + 2)

    # Plot histogram with aligned bars
    plt.hist(
        prop.histF,
        bins=bins,
        align="mid",
        rwidth=0.7,
        color="gray",
        alpha=0.6,
        log=log,
    )
    plt.hist(
        prop.trim_histF,
        bins=bins,
        align="mid",
        rwidth=0.7,
        color="forestgreen",
        edgecolor="black",
        label="Trimmed",
        log=log,
    )

    # Set ticks at the centers of bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, labels=[str(int(c)) for c in bin_centers])

    plt.xlabel("Frequency")
    plt.ylabel("Counts (log scale)" if log else "Counts")
    plt.legend()
    plt.show()


def plot_obs2D(prop, cmap=None, log=False):
    # Data
    histW, histF = prop.histW, prop.histF
    trimW, trimF = prop.trim_histW, prop.trim_histF

    binsW = np.arange(max(max(histW), max(trimW)) + 2)
    binsF = np.arange(max(max(histF), max(trimF)) + 2)

    H_full, xedges, yedges = np.histogram2d(histW, histF, bins=[binsW, binsF])
    H_trim, _, _ = np.histogram2d(trimW, trimF, bins=[binsW, binsF])

    # Colormaps
    base_cmap = plt.cm.cool
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[0, -1] = 0.0
    cool_cmap = LinearSegmentedColormap.from_list("cool_transparent", colors)
    soft_greys = LinearSegmentedColormap.from_list(
        "soft_greys",
        [(0.0, (0.5, 0.5, 0.5, 0.0)), (1.0, (0.5, 0.5, 0.5, 1.0))],
        N=256,
    )

    # GridSpec: plot + narrow colorbar for each
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 0.02, 1, 0.02])

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    cbar_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])]

    datasets = [H_full, H_trim]
    titles = ["Full", "Effective"]
    cmaps = [soft_greys, cool_cmap if cmap is None else cmap]

    for i, (ax, H, title, cax, cmap_ax) in enumerate(
        zip(axes, datasets, titles, cbar_axes, cmaps)
    ):
        im = ax.imshow(
            H.T,
            origin="lower",
            cmap=cmap_ax,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",  # ensures the plot fills the axes
        )
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_title(title)
        ax.set_xlabel("Pauli weight")
        if i == 0:
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel("")

        # colorbar tightly next to the plot
        plt.colorbar(
            im, cax=cax, label="Counts (log scale)" if log else "Counts", pad=-2
        )

    # remove all extra padding
    plt.subplots_adjust(left=0.05, right=1, top=0.9, bottom=0.1)
    plt.show()
