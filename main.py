import numpy as np
from utils.paths import get_subjects
from utils.graphs import GraphFromCSV, delete_rois
from multiprocessing import cpu_count, Process
import networkx as nx

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
    stroke_followup_2_paths = [path for path in files if '/Stroke/ses-followup-2' in path]
    # stroke_followup2_paths = [path for path in files if '/Stroke/ses-followup2' in path]     # followup2 is after 1 year
    glioma_preop_paths = [path for path in files if '/Glioma/ses-preop/' in path]
    glioma_postop_paths = [path for path in files if '/Glioma/ses-postop/' in path]
    glioma_control_paths = [path for path in files if '/Glioma/ses-control/' in path]

    stroke_acute_adj = create_tensor_from_multiple_adjacency_graphs(stroke_acute_paths, 'stroke_acute')
    stroke_control_adj = create_tensor_from_multiple_adjacency_graphs(stroke_control_paths, 'stroke_control')
    stroke_followup_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup_paths, 'stroke_followup')
    stroke_followup_2_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup_2_paths, 'stroke_followup_2')

    glioma_preop_adj = create_tensor_from_multiple_adjacency_graphs(glioma_preop_paths, 'glioma_preop')
    glioma_postop_adj = create_tensor_from_multiple_adjacency_graphs
    (glioma_postop_paths, 'glioma_postop')
    glioma_control_adj = create_tensor_from_multiple_adjacency_graphs
    (glioma_control_paths, 'glioma_control')

    mods_acute, mods_followp, mods_followp_2 = [], [], []
    locef_acute, locef_followp, locef_followp_2 = [], [], []
    gloef_acute, gloef_followp, gloef_followp_2 = [], [], []
    for i in range(stroke_acute_adj.shape[-1]):
        G = nx.from_numpy_array(stroke_acute_adj[...,0])
        communities = nx.algorithms.community.louvain_communities(G)
        mods_acute.append(nx.algorithms.community.modularity(G, communities))
    mods_acute = np.array(mods_acute)
    for i in range(stroke_followup_adj.shape[-1]):
        G = nx.from_numpy_array(stroke_followup_adj[...,0])
        communities = nx.algorithms.community.louvain_communities(G)
        mods_followp.append(nx.algorithms.community.modularity(G, communities))
    mods_followp = np.array(mods_followp)
    for i in range(stroke_followup_2_adj.shape[-1]):
        G = nx.from_numpy_array(stroke_followup_2_adj[...,0])
        communities = nx.algorithms.community.louvain_communities(G)
        mods_followp_2.append(nx.algorithms.community.modularity(G, communities))
    mods_followup_2 = np.array(mods_followp_2)

    print(mods_acute, mods_followp, mods_followp_2)
    # stroke_followup2_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup2_paths, 'stroke_followup2')

    """ procs = []

    experiments = [[stroke_acute_adj, stroke_control_adj, 'stoke_acute vs stroke_control'],
                   [glioma_preop_adj, glioma_control_adj, 'glioma preop vs glioma control'],
                   [stroke_acute_adj, glioma_preop_adj, 'stroke acute vs glioma preop'],
                   [glioma_preop_adj, glioma_postop_adj, 'glioma preop vs glioma postop'],
                   [glioma_postop_adj, glioma_control_adj, 'glioma postop vs glioma control'],
                   [stroke_followup_adj, stroke_control_adj, 'stroke followup vs stroke control'],
                   [stroke_followup_adj, stroke_acute_adj, 'stroke followup vs stroke acute']]
    for exp in experiments:
        procs.append(Process(target=run, args=(exp[0], exp[1], exp[2])))
        procs[-1].start()

    # procs.append(Process(target=run, args=(stroke_acute_adj, stroke_control_adj, 'stoke_acute vs stroke_control')))
    # procs[-1].start()
    # Do it for all combinations we want to test

    for p in procs:
        p.join()
     """
    # Access tests with the name of the test and filter the t-stat matrix at various levels