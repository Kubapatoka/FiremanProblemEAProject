import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter, FuncAnimation
from IPython.core.display import HTML
import copy
from tqdm import tqdm


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
        self.fps = kwargs.get("fps", 0.3)

    def _pick_k(self, perm, k, unavailable):
        valid = [i for i in range(len(perm)) if perm[i] not in unavailable]
        return perm[valid[:k]]

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

    def _place_firegfighters_incremental(self, graph: nx.Graph, priority, num_teams):
        valid = [
            i for i in graph.nodes if graph.nodes[i]["on_fire"] or graph.nodes[i]["burned"] or graph.nodes[i]["guarded"]
        ]
        for node in priority[valid[:num_teams]]:
            graph.nodes[node]["guarded"] = True

    def _is_fire_active(self, graph: nx.Graph):
        return any(graph.nodes[node]["on_fire"] for node in graph.nodes)

    def _draw_graph(self, graph: nx.Graph, pos):
        colors = []
        for node in graph.nodes:
            if graph.nodes[node]["starting"]:
                colors.append(self.node_colors["starting"])
            elif graph.nodes[node]["guarded"]:
                colors.append(self.node_colors["guarded"])
            elif graph.nodes[node]["burned"]:
                colors.append(self.node_colors["burned"])
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

    def _simulate_fire(self, graph, output_path):
        # Compute layout once
        pos = nx.spring_layout(graph)

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

        progress_bar = tqdm(
            total=len(frames),
            desc="Rendering Frames",
            unit="frame",
            leave=False,
        )

        def update(frame):
            progress_bar.update(1)
            return ax.imshow(frames[frame])

        ani = FuncAnimation(fig, update, frames=len(frames))

        if output_path is None:
            html = HTML(ani.to_jshtml())
            progress_bar.close()
            return html
        else:
            ani.save(output_path, writer=PillowWriter(fps=self.fps))
            progress_bar.close()

    def _simulate_fire_lite(self, graph: nx.Graph, output_path):
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

            nodes_to_delete = []
            for node in graph.nodes:
                if graph.nodes[node]["burned"] == True:
                    nodes_to_delete.append(node)

            for node in nodes_to_delete:
                graph.remove_node(node)
            self._update_fire_state(graph)

        # Add a few frames of the final state
        for _ in range(self.frames_after_fire_done):
            frames.append(frames[-1])

        progress_bar = tqdm(
            total=len(frames),
            desc="Rendering Frames",
            unit="frame",
            leave=False,
        )

        def update(frame):
            progress_bar.update(1)
            return ax.imshow(frames[frame])

        ani = FuncAnimation(fig, update, frames=len(frames))

        if output_path is None:
            html = HTML(ani.to_jshtml())
            progress_bar.close()
            return html
        else:
            ani.save(output_path, writer=PillowWriter(fps=self.fps))
            progress_bar.close()

    def simulate_fire_incremental(self, graph, fire_starts, num_teams, firefighter_placement, **kwargs):
        # Deep copy the graph to avoid modifying the original
        graph_copy = copy.deepcopy(graph)

        # Initialize attributes
        for node in graph_copy.nodes:
            graph_copy.nodes[node]["burned"] = False
            graph_copy.nodes[node]["on_fire"] = node in fire_starts
            graph_copy.nodes[node]["starting"] = node in fire_starts
        self._place_firegfighters_incremental(graph, firefighter_placement, num_teams)

        # Compute layout once
        pos = nx.spring_layout(graph)

        # Initialize the GIF frames
        frames = []
        fig, ax = plt.subplots()

        # Simulation loop
        it = 0
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

            if kwargs.get('delete_burned', False):
                nodes_to_delete = []
                for node in graph.nodes:
                    if graph.nodes[node]["burned"] == True:
                        nodes_to_delete.append(node)

                for node in nodes_to_delete:
                    graph.remove_node(node)

            if it % 2 == 0:
                self._update_fire_state(graph)
            else:
                self._place_firegfighters_incremental(graph, firefighter_placement, num_teams)

        # Add a few frames of the final state
        for _ in range(self.frames_after_fire_done):
            frames.append(frames[-1])

        progress_bar = tqdm(
            total=len(frames),
            desc="Rendering Frames",
            unit="frame",
            leave=False,
        )

        def update(frame):
            progress_bar.update(1)
            return ax.imshow(frames[frame])

        ani = FuncAnimation(fig, update, frames=len(frames))

        if output_path is None:
            html = HTML(ani.to_jshtml())
            progress_bar.close()
            return html
        else:
            ani.save(output_path, writer=PillowWriter(fps=self.fps))
            progress_bar.close()

    def simulate_fire(
        self,
        graph,
        fire_starts,
        firefighter_placement,
        **kwargs,
    ):
        # Deep copy the graph to avoid modifying the original
        graph_copy = copy.deepcopy(graph)

        # Initialize attributes
        for node in graph_copy.nodes:
            graph_copy.nodes[node]["guarded"] = node in firefighter_placement
            graph_copy.nodes[node]["burned"] = False
            graph_copy.nodes[node]["on_fire"] = node in fire_starts
            graph_copy.nodes[node]["starting"] = node in fire_starts

        output_path = kwargs.get("output_path", None)
        return self._simulate_fire(graph_copy, output_path)

    def simulate_fire_lite(
        self,
        graph,
        fire_starts,
        firefighter_placement,
        **kwargs,
    ):
        # Deep copy the graph to avoid modifying the original
        graph_copy = copy.deepcopy(graph)

        # Initialize attributes
        for node in graph_copy.nodes:
            graph_copy.nodes[node]["guarded"] = node in firefighter_placement
            graph_copy.nodes[node]["burned"] = False
            graph_copy.nodes[node]["on_fire"] = node in fire_starts
            graph_copy.nodes[node]["starting"] = node in fire_starts

        output_path = kwargs.get("output_path", None)
        return self._simulate_fire_lite(graph_copy, output_path)


    def simulate_multiple_fireman_scenarios(
        self,
        graph,
        fire_starts,
        firefighter_placements,
        **kwargs,
    ):
        pos = nx.spring_layout(graph, seed=42)
        fig, ax = plt.subplots()

        total_frames = len(firefighter_placements)
        progress_bar = tqdm(
            total=total_frames,
            desc="Rendering Frames",
            unit="frame",
            leave=False,
        )

        def update(idx):
            fireman, _ = firefighter_placements[idx]
            progress_bar.update(1)
            graph_copy = copy.deepcopy(graph)
            for node in graph_copy.nodes:
                graph_copy.nodes[node]["guarded"] = node in fireman
                graph_copy.nodes[node]["burned"] = False
                graph_copy.nodes[node]["on_fire"] = node in fire_starts
                graph_copy.nodes[node]["starting"] = node in fire_starts

            while self._is_fire_active(graph_copy):
                self._update_fire_state(graph_copy)

            ax.clear()
            self._draw_graph(graph_copy, pos)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            return ax.imshow(frame)

        ani = FuncAnimation(fig, update, frames=total_frames)

        output_path = kwargs.get("output_path", None)
        if output_path is None:
            html = HTML(ani.to_jshtml())
            progress_bar.close()
            return html
        else:
            ani.save(output_path, writer=PillowWriter(fps=self.fps))
            plt.close(fig)
            progress_bar.close()
            print(f"GIF saved to {output_path}")
