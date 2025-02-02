import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter, FuncAnimation
import copy


class Displayer:
    def __init__(self, **kwargs):
        self.node_colors = kwargs.get(
            "node_colors",
            {
                "guarded": "blue",
                "burned": "brown",
                "on_fire": "red",
                "starting": "yellow",
                "default": "green",
            },
        )
        self.frames_after_fire_done = kwargs.get("frames_after_fire_done", 3)
        self.fps = kwargs.get("fps", 1)

    def _update_fire_state(self, graph: nx.Graph):
        new_on_fire = []
        for node in graph.nodes:
            if graph.nodes[node]["on_fire"]:
                graph.nodes[node]["burned"] = True
                graph.nodes[node]["on_fire"] = False

                # Spread fire to neighbors
                for neighbor in graph.neighbors(node):
                    if (
                        not graph.nodes[neighbor]["burned"]
                        and not graph.nodes[neighbor]["on_fire"]
                        and not graph.nodes[neighbor]["guarded"]
                    ):
                        new_on_fire.append(neighbor)

        # Set new nodes on fire
        for node in new_on_fire:
            graph.nodes[node]["on_fire"] = True

    def _is_fire_active(self, graph: nx.Graph):
        return any(graph.nodes[node]["on_fire"] for node in graph.nodes)

    def _draw_graph(self, graph: nx.Graph, pos):
        colors = []
        for node in graph.nodes:
            if graph.nodes[node]["guarded"]:
                colors.append(self.node_colors["guarded"])
            elif graph.nodes[node]["burned"]:
                colors.append(self.node_colors["burned"])
            elif graph.nodes[node]["starting"]:
                colors.append(self.node_colors["starting"])
            elif graph.nodes[node]["on_fire"]:
                colors.append(self.node_colors["on_fire"])
            else:
                colors.append(self.node_colors["default"])

        nx.draw(
            graph,
            pos=pos,
            with_labels=True,
            node_color=colors,
            node_size=800,
            font_size=10,
        )

    def _simulate_fire(self, graph, gif_path):
        # Compute layout once
        pos = nx.spring_layout(graph, seed=42)

        # Initialize the GIF frames
        frames = []
        fig, ax = plt.subplots()

        # Simulation loop
        while True:
            ax.clear()
            self._draw_graph(graph, pos)

            # Capture frame
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

            if not self._is_fire_active(graph):
                break

            self._update_fire_state(graph)

        # Add a few frames of the final state
        for _ in range(self.frames_after_fire_done):
            frames.append(frames[-1])

        # Save to GIF
        writer = PillowWriter(fps=self.fps)
        ani = FuncAnimation(fig, lambda i: ax.imshow(frames[i]), frames=len(frames))
        ani.save(gif_path, writer=writer)

    def simulate_fire(
        self,
        graph,
        fire_starts,
        firefighter_placement,
        gif_path="ourput/fire_simulation.gif",
    ):
        # Deep copy the graph to avoid modifying the original
        graph_copy = copy.deepcopy(graph)

        # Initialize attributes
        for node in graph_copy.nodes:
            graph_copy.nodes[node]["guarded"] = node in firefighter_placement
            graph_copy.nodes[node]["burned"] = False
            graph_copy.nodes[node]["on_fire"] = node in fire_starts

        self._simulate_fire(graph_copy, gif_path)

    def simulate_multiple_fireman_scenarios(
        self,
        graph,
        fire_starts,
        firefighter_placements,
        gif_path="fireman_scenarios.gif",
    ):
        """
        Simulates different fireman placements and generates a GIF showing the end result for each scenario.

        Args:
          graph (networkx.Graph): The input graph without attributes.
          fire_starts (list): List of nodes where fire starts.
          fireman_list (list of lists): Each inner list contains fireman placements for a scenario.
          gif_path (str): Path to save the output GIF.
        """
        pos = nx.spring_layout(graph, seed=42)
        frames = []
        fig, ax = plt.subplots()

        for fireman in firefighter_placements:
            graph_copy = copy.deepcopy(graph)
            for node in graph_copy.nodes:
                graph_copy.nodes[node]["guarded"] = node in fireman
                graph_copy.nodes[node]["burned"] = False
                graph_copy.nodes[node]["on_fire"] = node in fire_starts

            while self._is_fire_active(graph_copy):
                self._update_fire_state(graph_copy)

            ax.clear()
            self._draw_graph(graph_copy, pos)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

        for _ in range(self.frames_after_fire_done):
            frames.append(frames[-1])

        writer = PillowWriter(fps=self.fps)
        ani = FuncAnimation(fig, lambda i: ax.imshow(frames[i]), frames=len(frames))
        ani.save(gif_path, writer=writer)
