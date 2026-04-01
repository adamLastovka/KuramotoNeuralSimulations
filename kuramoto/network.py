''' Network analysis functions'''
import networkx as nx
from .coupling import CouplingMatrix
from .simulation import Simulation

import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


def create_cortical_graph(source: CouplingMatrix | jnp.ndarray | Simulation, omega: jnp.ndarray | None = None) -> nx.Graph:
    """Create a graph from a coupling matrix.
    Args:
        source: Coupling matrix, Simulation object, or numpy array. [N,N]
        omega: Natural frequencies (optional). [N,]
    Returns:
        NetworkX graph.
    """
    if isinstance(source, Simulation):
        K = source.coupling.K
        omega = source.params.omega
    elif isinstance(source, CouplingMatrix):
        K = source.K
    elif isinstance(source, jnp.ndarray):
        K = source
    else:
        raise ValueError(f"Unknown source object type: {type(source)}")

    N = K.shape[0]

    graph = nx.from_numpy_array(K)

    # Add node attributes
    if omega is not None:
        for i in range(N):
            graph.nodes[i]["natural_frequency"] = omega[i]

    if hasattr(source, "coupling") and source.coupling.group_ids is not None:
        for i in range(N):
            graph.nodes[i]["group"] = source.coupling.group_ids[i]

    return graph

# --- Metrics ---
def get_degree(graph: nx.Graph) -> np.ndarray:
    """Get the degrees of all nodes in the graph.
    Args:
        graph: NetworkX graph.
    Returns:
        Degree vector. (N,) np.ndarray
    """
    return np.array([graph.degree(i) for i in graph.nodes()])

def get_deg_centrality(graph: nx.Graph) -> np.ndarray:
    """Get the degree centrality of all nodes in the graph.
    Args:
        graph: NetworkX graph.
    Returns:
        Degree centrality. (N,) np.ndarray
    """
    dc = nx.degree_centrality(graph)
    return np.fromiter(dc.values(), dtype=float)

def get_closeness_centrality(graph: nx.Graph) -> np.ndarray:
    """Compute the closeness centrality of a graph.
    Args:
        graph: NetworkX graph.
    Returns:
        Closeness centrality. (N,) np.ndarray
    """
    cc = nx.closeness_centrality(graph)
    return np.fromiter(cc.values(), dtype=float)
    

def get_betweenness_centrality(graph: nx.Graph) -> np.ndarray:
    """Compute the betweenness centrality of a graph.
    Args:
        graph: NetworkX graph.
    Returns:
        Betweenness centrality. (N,) np.ndarray
    """
    bc = nx.betweenness_centrality(graph)
    return np.fromiter(bc.values(), dtype=float)

def get_eigenvector_centrality(graph: nx.Graph) -> np.ndarray:
    """Compute the eigenvector centrality of a graph.
    Args:
        graph: NetworkX graph.
    Returns:
        Eigenvector centrality. (N,) np.ndarray
    """
    ec = nx.eigenvector_centrality(graph, weight="weight")
    return np.fromiter(ec.values(), dtype=float)

def get_graph_metrics(G: nx.Graph) -> dict[str, np.ndarray]:
    """Get the graph metrics.
    Args:
        G: NetworkX graph.
    Returns:
        Dictionary of metrics.
        - deg_cent: Degree centrality. (N,) np.ndarray
        - closeness: Closeness centrality. (N,) np.ndarray
        - betweenness: Betweenness centrality. (N,) np.ndarray
        - eigenvector: Eigenvector centrality. (N,) np.ndarray
    """
    return {
        "deg_cent": get_deg_centrality(G),
        "closeness": get_closeness_centrality(G),
        "betweenness": get_betweenness_centrality(G),
        "eigenvector": get_eigenvector_centrality(G),
    }

# --- Visualization ---
def plot_cortical_graph(graph: nx.Graph,layout: str = "spring", ax: plt.Axes | None = None) -> None:
    """Plot a cortical graph.
    Args:
        graph: NetworkX graph.
        layout: Layout algorithm. Options: "spring", "circular", "planar", "random", "grid".
        ax: Matplotlib axes.
    Returns:
        Matplotlib figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    if layout == "spring":
        pos = nx.spring_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "planar":
        pos = nx.planar_layout(graph)
    elif layout == "random":
        pos = nx.random_layout(graph)
    elif layout == "grid":
        pos = {}
        N = graph.number_of_nodes()

        if jnp.sqrt(N) % 1 != 0:
            raise ValueError("Grid layout doesnt support non-square grids for now")

        for i in range(N):
            pos[i] = (i % int(jnp.sqrt(N)), i // int(jnp.sqrt(N))) # Assumes square grid

    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Set edge size based on coupling strength
    edge_widths = np.array([graph.edges[e]["weight"] for e in graph.edges()])
    edge_widths = edge_widths / edge_widths.max() * 1 # Scale to 0-1
    nx.draw_networkx_edges(graph, pos=pos, width=edge_widths, ax=ax)

    # Set node color based on group membership
    if "group" in graph.nodes[0]:
        cmap = plt.get_cmap("tab10")
        num_groups = int(max(graph.nodes[i]["group"] for i in graph.nodes())) + 1

        node_colors = [cmap(graph.nodes[i]["group"] / num_groups) for i in graph.nodes()]

        nx.draw_networkx_nodes(graph, pos=pos, node_color=node_colors, ax=ax)
    else:
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax)
    
    # Set node labels based on node index
    node_labels = {i: i for i in graph.nodes()}
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels, ax=ax)

    # nx.draw(graph, pos=pos, with_labels=True, ax=ax)
    return fig, ax

def plot_graph_metrics(metrics: dict[str, np.ndarray] | None = None, G: nx.Graph | None = None, grid_shape: tuple[int, int]=None, title: str = "Network Metrics"):
    if metrics is None and G is None:
        raise ValueError("Either metrics or G must be provided")
    elif G is not None:
        metrics = get_graph_metrics(G)
    else:
        pass

    if grid_shape is None:
        raise ValueError("Grid shape must be provided")

    fig,ax = plt.subplots(1,4,figsize=(12,3),constrained_layout=True)
    ax = ax.ravel()

    titles = ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "Eigenvector Centrality"]
    for i, (metric, title_i) in enumerate(zip(metrics.values(), titles)):
        if np.max(metric) - np.min(metric) < 0.001: # uniform case
            norm=Normalize(vmin=0,vmax=1)
        else:
            norm=Normalize(vmin=0,vmax=np.max(metric))
        im = ax[i].imshow(metric.reshape(grid_shape),norm=norm)
        ax[i].set_title(title_i)
        fig.colorbar(im,ax=ax[i],fraction=0.046, pad=0.04, norm=norm)
    fig.suptitle(title)