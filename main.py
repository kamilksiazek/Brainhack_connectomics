
import numpy as np
from utils.paths import get_subjects
from utils.graphs import GraphFromCSV, delete_rois
from multiprocessing import cpu_count, Process
from tqdm import tqdm

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

def run(x, y, test_name):
    p_val, adj_edges, comp, t_test = nbs_bct(
        x,
        y,
        thresh=0.05
    )
    tests[test_name] = [p_val, adj_edges, comp, t_test]
    return 


if __name__ == '__main__':
    files = get_subjects('./Data/')
    cpus = cpu_count()
    tests = {}

    stroke_acute_paths = [path for path in files if '/Stroke/ses-acute/' in path]
    stroke_control_paths = [path for path in files if '/Stroke/ses-control' in path]
    stroke_followup_paths = [path for path in files if '/Stroke/ses-followup' in path]
    stroke_followup2_paths = [path for path in files if '/Stroke/ses-followup2' in path]     # followup2 is after 1 year
    
    stroke_acute_adj = create_tensor_from_multiple_adjacency_graphs(stroke_acute_paths, 'stroke_acute')
    stroke_control_adj = create_tensor_from_multiple_adjacency_graphs(stroke_control_paths, 'stroke_control')
    stroke_followup_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup_paths, 'stroke_followup')
    stroke_followup2_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup2_paths, 'stroke_followup2')


    procs = []

    procs.append(Process(target=run, args=(stroke_acute_adj, stroke_control_adj, 'stoke_acute vs stroke_control')))
    procs[-1].start()
    # Do it for all combinations we want to test

    for p in procs:
        p.join()
    
    # Access tests with the name of the test and filter the t-stat matrix at various levels