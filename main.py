
import numpy as np
from utils.paths import get_subjects
from utils.graphs import GraphFromCSV, delete_rois
from bct.nbs import nbs_bct


def create_tensor_from_multiple_adjacency_graphs(paths, prefix_name):
    adjacencies = []
    for path in paths:
        graph = GraphFromCSV(path,
        f"{prefix_name}_{path[path.rfind('sub-'):path.rfind('.csv')]}")
        adjacency = graph.get_connections()
        adjacencies.append(adjacency)
    adjacencies, new_rois = delete_rois(np.asarray(adjacencies))
    adjacencies = np.transpose(adjacencies, (1, 2, 0))
    return adjacencies

if __name__ == '__main__':
    files = get_subjects('./Data/')
    stroke_acute_paths = [path for path in files if '/Stroke/ses-acute/' in path]
    stroke_control_paths = [path for path in files if '/Stroke/ses-control' in path]

    acute_adjacencies = create_tensor_from_multiple_adjacency_graphs(stroke_acute_paths, 'stroke_acute')
    control_adjacencies = create_tensor_from_multiple_adjacency_graphs(stroke_control_paths, 'stroke_control')

    p_val, adj_edges, comp = nbs_bct(
        acute_adjacencies,
        control_adjacencies,
        thresh=0.05
    )
