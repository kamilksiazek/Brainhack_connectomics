import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.paths import get_subjects
from utils.graphs import GraphFromCSV, delete_rois
from threading import Thread, Lock
from nbs import nbs_bct

lock = Lock()
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

class Test(Thread):
    def __init__(self, x, y, name, percentile):
        Thread.__init__(self)
        self.x = x
        self.y = y
        self.percentile = percentile
        self.name = name.replace(' ', '_')
        self.threshold = self.load_threshold()

    def load_threshold(self):
        percentiles = np.load(f'./Tables/percentiles_{self.name}.npy')
        # This table has 99 values: percentiles from 1 to 99.
        # For instance, 95th percentile is on the 94th position.
        return percentiles[self.percentile - 1]

    def run(self):
        p_val, adj_edges, comp = nbs_bct(
            self.x,
            self.y,
            thresh=self.threshold,
            verbose=True
        )
        lock.acquire()
        np.savez(f'./Results/{self.name}_{self.percentile}_percentile.npz',
                 p_val, adj_edges, comp)
        tests[self.name] = [self.percentile, self.threshold, p_val, adj_edges, comp]
        lock.release()
        return


def prepare_histogram_based_on_t_statistics(matrix, key):
    sns.set_style("whitegrid")
    plt.hist(matrix, density=True)
    plt.xlabel('t-statistics')
    plt.ylabel('Probability density')
    plt.xlim(0, 12)
    plt.ylim(0, 0.5)
    for percentile, color in zip([50, 75, 95], ['blue', 'red', 'green']):
        plt.axvline(x=np.percentile(matrix, percentile),
                    label=f'{percentile}. percentile',
                    color=color)
    key = key.replace(' ', '_')
    plt.title(key)
    plt.legend()
    plt.savefig(f'histogram_{key}.png', dpi=300)
    plt.close()


def calculate_and_plot_percentiles(matrix, key):
    sns.set_style("whitegrid")
    key = key.replace(' ', '_')
    percentiles = np.percentile(matrix, np.arange(1, 100, 1))
    np.save(f'percentiles_{key}.npy', percentiles)
    plt.xlabel('Percentile')
    plt.ylabel('t-statistics')
    plt.title(key)
    plt.plot(percentiles)
    plt.savefig(f'percentiles_{key}.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    files = get_subjects('./Data/')
    tests = {}
    os.makedirs('./Results/', exist_ok=True)

    stroke_acute_paths = [path for path in files if '/Stroke/ses-acute/' in path]
    stroke_control_paths = [path for path in files if '/Stroke/ses-control' in path]
    stroke_followup_paths = [path for path in files if '/Stroke/ses-followup' in path]
    stroke_followup_2_paths = [path for path in files if '/Stroke/ses-followup-2' in path]
    # stroke_followup2_paths = [path for path in files if '/Stroke/ses-followup2' in path]     # followup2 is after 1 year
    glioma_preop_paths = [path for path in files if '/Glioma/ses-preop' in path]
    glioma_postop_paths = [path for path in files if '/Glioma/ses-postop' in path]
    glioma_control_paths = [path for path in files if '/Glioma/ses-control' in path]

    stroke_acute_adj = create_tensor_from_multiple_adjacency_graphs(stroke_acute_paths, 'stroke_acute')
    stroke_control_adj = create_tensor_from_multiple_adjacency_graphs(stroke_control_paths, 'stroke_control')
    stroke_followup_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup_paths, 'stroke_followup')
    stroke_followup_2_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup_2_paths, 'stroke_followup_2')

    glioma_preop_adj = create_tensor_from_multiple_adjacency_graphs(glioma_preop_paths, 'glioma_preop')
    glioma_postop_adj = create_tensor_from_multiple_adjacency_graphs(glioma_postop_paths, 'glioma_postop')
    glioma_control_adj = create_tensor_from_multiple_adjacency_graphs(glioma_control_paths, 'glioma_control')

    # mods_acute, mods_followp, mods_followp_2 = [], [], []
    # locef_acute, locef_followp, locef_followp_2 = [], [], []
    # gloef_acute, gloef_followp, gloef_followp_2 = [], [], []
    # for i in range(stroke_acute_adj.shape[-1]):
    #     G = nx.from_numpy_array(stroke_acute_adj[...,0])
    #     communities = nx.algorithms.community.louvain_communities(G)
    #     mods_acute.append(nx.algorithms.community.modularity(G, communities))
    # mods_acute = np.array(mods_acute)
    # for i in range(stroke_followup_adj.shape[-1]):
    #     G = nx.from_numpy_array(stroke_followup_adj[...,0])
    #     communities = nx.algorithms.community.louvain_communities(G)
    #     mods_followp.append(nx.algorithms.community.modularity(G, communities))
    # mods_followp = np.array(mods_followp)
    # for i in range(stroke_followup_2_adj.shape[-1]):
    #     G = nx.from_numpy_array(stroke_followup_2_adj[...,0])
    #     communities = nx.algorithms.community.louvain_communities(G)
    #     mods_followp_2.append(nx.algorithms.community.modularity(G, communities))
    # mods_followup_2 = np.array(mods_followp_2)

    # print(mods_acute, mods_followp, mods_followp_2)
    # stroke_followup2_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup2_paths, 'stroke_followup2')

    procs = []

    experiments = [[stroke_acute_adj, stroke_control_adj, 'stroke_acute vs stroke_control'],
                   #[glioma_preop_adj, glioma_control_adj, 'glioma preop vs glioma control'],
                   #[stroke_acute_adj, glioma_preop_adj, 'stroke acute vs glioma preop'],
                   #[glioma_preop_adj, glioma_postop_adj, 'glioma preop vs glioma postop'],
                   #[glioma_postop_adj, glioma_control_adj, 'glioma postop vs glioma control'],
                   [stroke_followup_adj, stroke_control_adj, 'stroke followup vs stroke control'],
                   [stroke_followup_adj, stroke_acute_adj, 'stroke followup vs stroke acute'],
                   [stroke_followup_2_adj, stroke_control_adj, 'stroke followup2 vs stroke control'],
                   [stroke_followup_2_adj, stroke_acute_adj, 'stroke followup2 vs stroke acute']]
    for exp in experiments:
        for percentile in [50, 75, 95]:
            procs.append(Test(x=exp[0], y=exp[1], name=exp[2], percentile=percentile))
            procs[-1].start()

    # procs.append(Process(target=run, args=(stroke_acute_adj, stroke_control_adj, 'stoke_acute vs stroke_control')))
    # procs[-1].start()
    # Do it for all combinations we want to test

    for p in procs:
        p.join()
    
    # Access tests with the name of the test and filter the t-stat matrix at various levels

    # for key in tests.keys():
    #     triu_matrix = tests[key][0]
    #     square_matrix = np.reshape(triu_matrix, (1, triu_matrix.shape[0])) + np.reshape(triu_matrix, (triu_matrix.shape[0], 1))
    #     square_matrix = square_matrix.flatten()
    #     prepare_histogram_based_on_t_statistics(square_matrix, key)
    #     calculate_and_plot_percentiles(square_matrix, key)

    # Access tests with the name of the test and filter the t-stat matrix at various levels
