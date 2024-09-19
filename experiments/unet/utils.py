import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import numpy as np

from data.constants import ZONING_NAMES

def plot_polygon(ax, poly, label=None, **kwargs):
    x, y = poly.exterior.xy
    ax.fill(x, y, label=label, **kwargs)
    return

def plot_polygon_single(poly, label=None, **kwargs):
    fig = plt.figure()  # Create a new figure
    x, y = poly.exterior.xy
    plt.fill(x, y, label=label, **kwargs)  # plt.fill works on the current figure
    return

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
                               width=edge_size, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=efront, edge_color=c_edge[3],
                               width=edge_size, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, edge_color=c_edge[0],
                               width=edge_size, ax=ax)

    ax.axis('off')
    
def plot_graph_single(G, c_node='black', c_edge=['white']*4, dw_edge=False, pos=None, node_size=10,
               edge_size=1, title=None):

    plt.figure(figsize=(8, 6)) 
    
    # position
    if pos is None:
        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=c_node)

    # edges
    if dw_edge:
        epass = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'passage']
        edoor = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'door']
        efront = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'entrance']
        # red full for passage, red dashed for door, yellow dashed for front
        nx.draw_networkx_edges(G, pos, edgelist=epass, edge_color=c_edge[1],
                               width=edge_size)
        nx.draw_networkx_edges(G, pos, edgelist=edoor, edge_color=c_edge[2],
                               width=edge_size, style='dashed')
        nx.draw_networkx_edges(G, pos, edgelist=efront, edge_color=c_edge[3],
                               width=edge_size, style='dashed')
        # Create a legend
        plt.plot([], color=c_edge[1], label='Passage')
        plt.plot([], color=c_edge[2], label='Door', linestyle='dashed')
        plt.plot([], color=c_edge[3], label='Entrance', linestyle='dashed')
        plt.legend()
    else:
        nx.draw_networkx_edges(G, pos, edge_color=c_edge[0],
                               width=edge_size)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.show()  # display the graph


def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object


def colorize_floorplan(img, classes, cmap):

    """
    Colorizes an integer-valued image (multi-class segmentation mask)
    based on a pre-defined cmap colorset.
    """

    h, w = np.shape(img)
    img_c = (np.ones((h, w, 3)) * 255).astype(int)
    for cat in classes:
        color = np.array(cmap(cat))[:3] * 255
        img_c[img == cat, :] = (color).astype(int)

    return img_c