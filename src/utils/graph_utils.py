import numpy as np
import torch
from eap.graph import Graph


def extract_edge_scores(example_path: str) -> np.ndarray:
    """Load per-example edge attribution scores."""
    edge_graph = Graph.from_pt(example_path)
    scores = np.array([edge_graph.edges[e].score for e in edge_graph.edges])
    return scores


def extract_edge_names(example_path: str) -> list[str]:
    """Return ordered edge names matching extract_edge_scores() output.

    Args:
        example_path: Path to any .pt graph file (same model → same topology).

    Returns:
        List of edge name strings, length n_edges.
    """
    edge_graph = Graph.from_pt(example_path)
    return list(edge_graph.edges.keys())


def build_edge_index(ref_path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load a reference graph once and return edge names + index arrays for fast extraction.

    Creates the Graph object a single time to capture the (row, col) positions
    of each edge in the 2-D scores tensor.  Subsequent files with the same
    model topology can be read with load_scores_fast(), which skips
    Graph.from_model() entirely.

    Args:
        ref_path: Path to any .pt graph file for this model (same topology for
            all files from the same run).

    Returns:
        edge_names: ordered list of edge name strings.
        rows: 1-D int array of row indices into the scores tensor.
        cols: 1-D int array of col indices into the scores tensor.
    """
    ref_graph = Graph.from_pt(ref_path)
    edge_names = list(ref_graph.edges.keys())
    rows = np.array([ref_graph.edges[e].matrix_index[0] for e in edge_names], dtype=np.intp)
    cols = np.array([ref_graph.edges[e].matrix_index[1] for e in edge_names], dtype=np.intp)
    return edge_names, rows, cols


def load_scores_fast(pt_path: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Extract edge scores from a .pt file without constructing a Graph object.

    Uses precomputed row/col indices (from build_edge_index) to slice the
    raw scores tensor directly, avoiding the expensive Graph.from_model() call.

    Args:
        pt_path: Path to a .pt graph file.
        rows: Row indices returned by build_edge_index.
        cols: Col indices returned by build_edge_index.

    Returns:
        1-D float32 numpy array of edge scores, same ordering as edge_names
        from build_edge_index.
    """
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    scores_tensor = d["edges_scores"]  # shape (n_src, n_dst)
    return scores_tensor[rows, cols].numpy()