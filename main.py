
import numpy as np
import matplotlib.pyplot as plt
from utils.paths import get_subjects
from utils.graphs import GraphFromCSV, delete_rois
from threading import Thread, Lock
from nbs import nbs_bct

lock = Lock()

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
    def __init__(self, x, y, test):
        Thread.__init__(self)
        self.x = x
        self.y = y
        self.test = test

    def run(self):
        # p_val, adj_edges, comp, t_test = nbs_bct(
        t_test = nbs_bct(
            self.x,
            self.y,
            thresh=0.05
        )
        lock.acquire()
        tests[self.test] = [t_test]
        lock.release()
        return 


if __name__ == '__main__':
    files = get_subjects('./Data/')
    tests = {}

    stroke_acute_paths = [path for path in files if '/Stroke/ses-acute/' in path]
    stroke_control_paths = [path for path in files if '/Stroke/ses-control' in path]
    stroke_followup_paths = [path for path in files if '/Stroke/ses-followup' in path]
    # stroke_followup2_paths = [path for path in files if '/Stroke/ses-followup2' in path]     # followup2 is after 1 year
    glioma_preop_paths = [path for path in files if '/Glioma/ses-preop' in path]
    glioma_postop_paths = [path for path in files if '/Glioma/ses-postop' in path]
    glioma_control_paths = [path for path in files if '/Glioma/ses-control' in path]

    stroke_acute_adj = create_tensor_from_multiple_adjacency_graphs(stroke_acute_paths, 'stroke_acute')
    stroke_control_adj = create_tensor_from_multiple_adjacency_graphs(stroke_control_paths, 'stroke_control')
    stroke_followup_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup_paths, 'stroke_followup')

    glioma_preop_adj = create_tensor_from_multiple_adjacency_graphs(glioma_preop_paths, 'glioma_preop')
    glioma_postop_adj = create_tensor_from_multiple_adjacency_graphs(glioma_postop_paths, 'glioma_postop')
    glioma_control_adj = create_tensor_from_multiple_adjacency_graphs(glioma_control_paths, 'glioma_control')

    # stroke_followup2_adj = create_tensor_from_multiple_adjacency_graphs(stroke_followup2_paths, 'stroke_followup2')

    procs = []

    experiments = [[stroke_acute_adj, stroke_control_adj, 'stoke_acute vs stroke_control'],
                   [glioma_preop_adj, glioma_control_adj, 'glioma preop vs glioma control'],
                   [stroke_acute_adj, glioma_preop_adj, 'stroke acute vs glioma preop'],
                   [glioma_preop_adj, glioma_postop_adj, 'glioma preop vs glioma postop'],
                   [glioma_postop_adj, glioma_control_adj, 'glioma postop vs glioma control'],
                   [stroke_followup_adj, stroke_control_adj, 'stroke followup vs stroke control'],
                   [stroke_followup_adj, stroke_acute_adj, 'stroke followup vs stroke acute']]
    for exp in experiments:
        procs.append(Test(x=exp[0], y=exp[1], test=exp[2]))
        procs[-1].start()

    # procs.append(Process(target=run, args=(stroke_acute_adj, stroke_control_adj, 'stoke_acute vs stroke_control')))
    # procs[-1].start()
    # Do it for all combinations we want to test

    for p in procs:
        p.join()
    
    # Access tests with the name of the test and filter the t-stat matrix at various levels

    for key in tests.keys():
        triu_matrix = tests[key][0]
        square_matrix = np.reshape(triu_matrix, (1, triu_matrix.shape[0])) + np.reshape(triu_matrix, (triu_matrix.shape[0], 1))
        np.save(f'{key}.npy', square_matrix)
        plt.imshow(square_matrix)
        plt.colorbar()
        plt.title(key)
        plt.savefig(f'{key}.pdf', dpi=300)
        plt.close()
