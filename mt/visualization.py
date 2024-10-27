import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.patches
from mt.constants import CMAP_MASK
import napari

sns.set_context("paper")
sns.set_theme(font="serif", style="dark", font_scale=1)

def visualize_region_properties(image: np.ndarray,
                                mean_areas: np.ndarray,
                                contact_percent: np.ndarray,
                                fig_width: int = 28,
                                font_scale=2):
    """Visualize the mean cell area and contact percentage of a 2D slice in a 3-panel figure.

    Args:
        image (np.ndarray): 2D image of the region.
        mean_areas (np.ndarray): 2D array of the mean cell areas.
        contact_percent (np.ndarray): 2D array of the contact percentage.
        fig_width (int, optional): Width of the figure. Defaults to 28.
        font_scale (int, optional): Font scale. Defaults to 2.

    Returns:
        plt.Figure: The figure.
    """
    sns.set_context("paper")
    sns.set_theme(font="serif", style="dark", font_scale=font_scale)
    h, w = image.shape
    fig, axs = plt.subplots(1, 3, figsize=(fig_width, 5))
    fig.subplots_adjust(wspace=0)
    cmap = CMAP_MASK

    def set_ticks_image(ax, h, w, image):
        major_ticks_x = [ii * w / 5 for ii in range(5)]
        minor_ticks_x = [ii * w / 5 + w / 10 for ii in range(5)]
        major_ticks_y = [ii * h / 5 for ii in range(5)]
        minor_ticks_y = [ii * h / 5 + h / 10 for ii in range(5)]
        minor_ticklabels = [str(ii) for ii in range(1, 6)]
        major_ticklabels = [""] * 5

        if image:
            ax.set_xticks(major_ticks_x)
            ax.set_xticks(minor_ticks_x, minor=True)

            ax.set_yticks(major_ticks_y)
            ax.set_yticks(minor_ticks_y, minor=True)
        else:
            ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
            ax.set_xticks([ii for ii in range(5)], minor=True)
            ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
            ax.set_yticks([ii for ii in range(5)], minor=True)

        ax.set_xticklabels(major_ticklabels)
        ax.set_xticklabels(minor_ticklabels, minor=True)
        ax.set_yticklabels(major_ticklabels)
        ax.set_yticklabels(minor_ticklabels, minor=True)
        ax.grid(True)

    axs[0].imshow(image, cmap=cmap)
    set_ticks_image(axs[0], h, w, True)
    axs[0].set_title("X-Y Slice")
    legend_elements = [
        # matplotlib.patches.Patch(facecolor=[0.878] * 3, label="X-Y Slice: ", edgecolor=[0.878] * 3),
        matplotlib.patches.Patch(facecolor=cmap(0), label="Air"),
        matplotlib.patches.Patch(facecolor=cmap(1), label="Polymer"),
        matplotlib.patches.Patch(facecolor=cmap(2), label="Al")]
    # box = axs[0].get_position()
    # axs[0].set_position([box.x0, box.y0-0.1, box.width, box.height])

    axs[0].legend(handles=legend_elements,
                  loc='upper center',
                  bbox_to_anchor=(0.5, 1.2),
                  ncols=3,
                  facecolor=[0.878] * 3,
                  framealpha=1)

    im = axs[1].imshow(mean_areas, cmap="RdYlGn_r", vmin=0, vmax=0.05)
    axs[1].set_title("Mean area of cells in region in $\mathrm{mm^2}$")
    set_ticks_image(axs[1], h, w, False)
    fig.colorbar(im, ax=axs[1])
    axs[1].set_aspect(h / w)

    im = axs[2].imshow(contact_percent * 100, cmap="RdYlGn", vmin=0, vmax=100)
    axs[2].set_title("Al-PMMA contact percentage")
    set_ticks_image(axs[2], h, w, False)
    fig.colorbar(im, ax=axs[2])
    axs[2].set_aspect(h / w)
    return fig

def show_in_napari(img, *labels):
    """Show an image and labels in napari."""
    viewer = napari.Viewer()
    viewer.add_image(img)
    for label in labels:
        if label is not None:
            viewer.add_labels(label)


def visualize_region_correlation(areas: np.ndarray,
                                 contact: np.ndarray,):
    sns.set_context("paper")
    sns.set_theme(font="serif", style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(1, figsize=(10, 5))
    axs.scatter(areas.ravel(), contact.ravel()*100, c=areas.ravel(), cmap="plasma")
    axs.set_title("Polymer contact percentage vs mean voronoi cell area for a 5x5 grid")
    axs.set_xlabel("Mean area of cells in region in $\mathrm{mm^2}$")
    axs.set_xlim(0, 0.1)
    axs.set_ylabel("Al-Polymer contact in %")
    axs.set_ylim(-5, 105)
    return fig