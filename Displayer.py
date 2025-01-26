import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter, FuncAnimation

class Displayer:
  def __init__(self, node_colors=None, frames_after_fire_done=3):
    self.node_colors = node_colors or {
      "guarded": "blue",
      "burned": "black",
      "on_fire": "red",
      "default": "green",
    }
    self.frames_after_fire_done = frames_after_fire_done

  def update_fire_state(self, graph: nx.Graph):
    new_on_fire = []
    for node in graph.nodes:
      if graph.nodes[node]["on_fire"]:
        graph.nodes[node]["burned"] = True
        graph.nodes[node]["on_fire"] = False

        # Spread fire to neighbors
        for neighbor in graph.neighbors(node):
          if (
            not graph.nodes[neighbor]["burned"] and
            not graph.nodes[neighbor]["on_fire"] and
            not graph.nodes[neighbor]["guarded"]
          ):
            new_on_fire.append(neighbor)

    # Set new nodes on fire
    for node in new_on_fire:
      graph.nodes[node]["on_fire"] = True

  def is_fire_active(self, graph: nx.Graph):
    return any(graph.nodes[node]["on_fire"] for node in graph.nodes)

  def draw_graph(self, graph: nx.Graph):
    colors = []
    for node in graph.nodes:
      if graph.nodes[node]["guarded"]:
        colors.append(self.node_colors["guarded"])
      elif graph.nodes[node]["burned"]:
        colors.append(self.node_colors["burned"])
      elif graph.nodes[node]["on_fire"]:
        colors.append(self.node_colors["on_fire"])
      else:
        colors.append(self.node_colors["default"])

    nx.draw(
      graph,
      pos=nx.spring_layout(graph),
      with_labels=True,
      node_color=colors,
      node_size=800,
      font_size=10,
    )

  def simulate_fire(self, graph, gif_path="fire_simulation.gif", fps=1):
    # Initialize the GIF frames
    frames = []
    fig, ax = plt.subplots()

    # Simulation loop
    while True:
      ax.clear()
      self.draw_graph(graph)

      # Capture frame
      fig.canvas.draw()
      frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      frames.append(frame)

      if not self.is_fire_active(graph):
        break

      self.update_fire_state(graph)

    # Add a few frames of the final state
    for _ in range(self.frames_after_fire_done):
      frames.append(frames[-1])

    # Save to GIF
    height, width, _ = frames[0].shape
    writer = PillowWriter(fps=fps)
    ani = FuncAnimation(fig, lambda i: ax.imshow(frames[i]), frames=len(frames))
    ani.save(gif_path, writer=writer)

