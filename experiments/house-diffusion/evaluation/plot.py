import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import networkx as nx

from constants import ZONING_NAMES


def plot_polygon(ax, poly, label=None, **kwargs):
    x, y = poly.exterior.xy
    ax.fill(x, y, label=label, **kwargs)
    return

def set_figure(nc, nr, fs):

    fig, axs = plt.subplots(ncols=nc, nrows=nr, figsize=(fs * nc, fs * nr))
    axs = axs.flatten() if nc * nr > 1 else axs

    return fig, axs


def plot_graph(G, ax, c_node='black', c_edge=['white']*4, dw_edge=False, pos=None, node_size=10,
               edge_size=10):

    """
    Plots the adjacency or access graph of a floor plan's corresponding graph structure.
    """

    # position
    if pos is None:
        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=c_node, ax=ax)

    # edges
    if dw_edge:
        epass = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'passage']
        edoor = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'door']
        efront = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'entrance']
        # red full for passage, red dashed for door, yellow dashed for front
        nx.draw_networkx_edges(G, pos, edgelist=epass, edge_color=c_edge[1],
                               width=edge_size, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edoor, edge_color=c_edge[2],
                               width=edge_size, style="dashed", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=efront, edge_color=c_edge[3],
                               width=edge_size, style="-.", ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, edge_color=c_edge[0],
                               width=edge_size, ax=ax)

    ax.axis('off')