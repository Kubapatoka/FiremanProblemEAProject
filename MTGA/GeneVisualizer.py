import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from time import time

class GeneEvolutionRenderer:
    """
    Render gene evolution as a GIF with bar graphs for each chromosome.

    Parameters:
        interval (int): Duration of each frame in milliseconds. Default is 500.
        fps (int): Frames per second for the GIF. Default is 2.
        figsize (tuple): Size of the figure for the plot. Default is (10, 5).
        bar_color (str): Color of the bars in the graph. Default is "blue".
    """
    def __init__(self, output_path: str, interval=500, fps=2, figsize=(10, 5), bar_color="blue", title="Gene Evolution - Generation {generation}"):
        self.interval = interval
        self.fps = fps
        self.figsize = figsize
        self.bar_color = bar_color
        self.output_path = output_path
        self.title_template = title


    def visualize(self, collected_data):
        """
        Generate a GIF showing the evolution of genes over generations.

        Parameters:
            collected_data (list[dict]): A list where each entry contains:
                - 'genes': np.ndarray of shape (num_of_tribes, num_of_chromosomes)
                - 'evaluations': list of floats, one for each tribe
            output_path (str): Path to save the GIF.
        """
        time0 = time()
        # # Extract scale limits for consistent bar graph scaling
        # max_value = max(data['genes'].max() for data in collected_data)
        # min_value = min(data['genes'].min() for data in collected_data)

        # Initialize figure
        num_tribes, num_chromosomes = collected_data[0]['genes'].shape
        fig, axes = plt.subplots(1, num_tribes, figsize=self.figsize)

        # Ensure axes is iterable even if there is only one tribe
        if num_tribes == 1:
            axes = [axes]

        # Initialize bar containers
        bar_containers = []
        for ax in axes:
            bars = ax.bar(range(num_chromosomes), np.zeros(num_chromosomes), color=self.bar_color)
            bar_containers.append(bars)

        def update(frame):
            # Get current data
            print(f"[{time() - time0:.6f}] handling frame {frame}")
            data = collected_data[frame]
            genes = data['genes']
            evaluations = data['evaluations']

            # Update figure title
            fig.suptitle(self.title_template.format(generation=frame), fontsize=16)

            # Update each subplot
            for tribe_idx, (ax, bars, gene, evaluation) in enumerate(zip(axes, bar_containers, genes, evaluations)):
                ax.clear()
                ax.bar(range(num_chromosomes), gene, color=self.bar_color)
                # ax.set_ylim(min_value, max_value)
                ax.set_title(f"Tribe {tribe_idx + 1}")
                ax.set_xlabel("")
                ax.set_ylabel(f"Value: {evaluation:.2f}")
                ax.set_xticks(range(num_chromosomes))

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(collected_data),
            interval=self.interval,
            blit=False
        )

        # Save the animation as a GIF
        anim.save(self.output_path, writer=PillowWriter(fps=self.fps))
        plt.close(fig)

        print(f"GIF saved to {self.output_path}")

    def visualize_independent(self, collected_data):
        """
        Generate a GIF showing the evolution of genes over generations.

        Parameters:
            collected_data (list[dict]): A list where each entry contains:
                - 'genes': np.ndarray of shape (num_of_tribes, num_of_chromosomes)
                - 'evaluations': list of floats, one for each tribe
            output_path (str): Path to save the GIF.
        """
        time0 = time()
        # Initialize figure
        num_tribes, num_chromosomes = collected_data[0]['genes'].shape
        fig, axes = plt.subplots(1, num_tribes, figsize=self.figsize)

        # Ensure axes is iterable even if there is only one tribe
        if num_tribes == 1:
            axes = [axes]

        # Initialize bar containers
        bar_containers = []
        for ax in axes:
            bars = ax.bar(range(num_chromosomes), np.zeros(num_chromosomes), color=self.bar_color)
            bar_containers.append(bars)

        # Set initial title
        fig.suptitle(self.title_template.format(generation=0), fontsize=16)

        def update(frame):
            # Get current data
            data = collected_data[frame]
            genes = data['genes']
            evaluations = data['evaluations']

            # Update figure title
            fig.suptitle(self.title_template.format(generation=frame), fontsize=16)

            # Update each subplot independently
            for tribe_idx, (ax, bars, gene, evaluation) in enumerate(zip(axes, bar_containers, genes, evaluations)):
                ax.clear()
                ax.bar(range(num_chromosomes), gene, color=self.bar_color)
                ax.set_ylim(gene.min() * 0.9, gene.max() * 1.1)  # Dynamic y-limits
                ax.set_title(f"Tribe {tribe_idx + 1}")
                ax.set_xlabel("")
                ax.set_ylabel(f"Value: {evaluation:.2f}")
                ax.set_xticks(range(num_chromosomes))

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(collected_data),
            interval=self.interval,
            blit=False
        )

        # Save the animation as a GIF
        anim.save(output_path, writer=PillowWriter(fps=self.fps))
        plt.close(fig)

        print(f"GIF saved to {self.output_path}")
