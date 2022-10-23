import networkx as nx
#import nxviz
import gudhi
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pandas as pd
from threading import Thread, Lock
from time import sleep

from utils.paths import get_subjects
from TDAlib import *
import os

plt.rcParams.update({
    "text.usetex": False
})
lineList = [line.rstrip('\n') for line in open('ROIs.txt')]
lock = Lock()
# Start with the curvature of each patient, 
# then make a distribution for each category and compare across timepoints

class Test(Thread):
    def __init__(self, matrix, name, persistences, log):
        Thread.__init__(self)
        self.matrix = matrix
        self.name = name
        self.persistences = persistences
        self.log = log

    def run(self):
        if self.log:
            self.matrix = np.log1p(self.matrix)
        self.matrix/= np.max(self.matrix)
        subj_name = self.name.split('_')[0]
        #os.makedirs(f'Plots/TDA/{subj_name}/', exist_ok=True)

        np.fill_diagonal(self.matrix, 0)
        Pdmatrix = pd.DataFrame(self.matrix)
        Pdmatrix.columns = lineList
        Pdmatrix.index = lineList
        Pdmatrix = Pdmatrix.sort_index(0).sort_index(1)
        mask = np.zeros_like(Pdmatrix.values, dtype=bool) 
        mask[np.triu_indices_from(mask)] = True 

        # This command will show you the functional connectivity matrix with Seaborn
        '''
        plt.figure(figsize = (20, 20))
        _ = sns.heatmap(np.log1p(self.matrix), cmap='coolwarm', cbar=True, square=False, mask=mask)
        plt.savefig(f'Plots/TDA/{subj_name}/{self.name}_masked.png')
        plt.close()
        '''
        # transform the connectivity matrix in a distance matrix
        self.matrix = np.ones_like(self.matrix) - self.matrix 
        rips_complex = gudhi.RipsComplex(distance_matrix=self.matrix)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        diag = simplex_tree.persistence()

        lock.acquire()
        self.persistences.extend(diag)
        lock.release()
        """ gudhi.plot_persistence_barcode(diag, legend=False, max_intervals=0)
        plt.savefig(f'Plots/TDA/{subj_name}/{name}_persistence_barcode.png')
        plt.close()

        gudhi.plot_persistence_diagram(diag, legend=True, max_intervals=0)
        plt.tick_params(axis='both', labelsize=15)
        plt.xlabel('Birth', fontsize=15)
        plt.ylabel('Death', fontsize=15)
        plt.savefig(f'Plots/TDA/{subj_name}/{name}_persistence_diagram.png')
        plt.close()

        gudhi.plot_persistence_density(diag, dimension=1)
        plt.savefig(f'Plots/TDA/{subj_name}/{name}_persistence_density.png')
        plt.close() """

        '''
        plotEuler_thr(matrix, 70)
        plt.savefig(f'Plots/TDA/{name}_euler_entrophy.png')
        plt.close()

        curvalues = Curv_thr(i=matrix, e=0.7)
        dict(zip(lineList[:10], curvalues[:10]))
        plt.figure(figsize=(20,5))
        sns.distplot(curvalues, kde=True, norm_hist=False)
        plt.xlabel('Curvature values')
        plt.ylabel('Counts')
        plt.savefig(f'Plots/TDA/{name}_curvature.png')
        plt.close()
        '''


def perform_analysis(population, population_name, log, threads=4):
    for subj in population:
        procs.append(Test(drop_data_in_connect_matrix(load_matrix(subj)), subj.split(os.sep)[-1].split('.')[0], persistences, log))
        procs[-1].start()
        while len(procs)>=threads:
            for p in procs:
                p.join(2)
                if not p.is_alive():
                    procs.remove(p)
                    break

    for p in procs:
        p.join()

    type = 'log1p' if log else 'original'

    plt.figure(figsize=(20,15))
    gudhi.plot_persistence_barcode(persistences, legend=False, max_intervals=0)
    plt.savefig(f'Plots/TDA/{population_name}_persistence_barcode_{type}.png')
    plt.close()

    plt.figure(figsize=(35,20))
    gudhi.plot_persistence_diagram(persistences, legend=True, max_intervals=0)
    plt.tick_params(axis='both', labelsize=15)
    plt.xlabel('Birth', fontsize=15)
    plt.ylabel('Death', fontsize=15)
    plt.savefig(f'Plots/TDA/{population_name}_persistence_diagram_{type}.png')
    plt.close()

    plt.figure(figsize=(20,15))
    gudhi.plot_persistence_density(persistences, dimension=1)
    plt.xlim([0,1]), plt.ylim([0,1])
    plt.savefig(f'Plots/TDA/{population_name}_persistence_density_dim-1_{type}.png')
    plt.close()

    plt.figure(figsize=(20,15))
    gudhi.plot_persistence_density(persistences, dimension=2)
    plt.xlim([0,1]), plt.ylim([0,1])
    plt.savefig(f'Plots/TDA/{population_name}_persistence_density_dim-2_{type}.png')
    plt.close()

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def drop_data_in_connect_matrix(connect_matrix, missing_labels=[35, 36, 81, 82]):
    index_to_remove = [(label - 1) for label in missing_labels]
    connect_matrix = np.delete(connect_matrix, index_to_remove, axis=0)
    connect_matrix = np.delete(connect_matrix, index_to_remove, axis=1) 
    return connect_matrix

files = get_subjects('./Data/')
os.makedirs('Plots/TDA/', exist_ok=True)

stroke_acute_paths = [path for path in files if '/Stroke/ses-acute/' in path]
stroke_control_paths = [path for path in files if '/Stroke/ses-control' in path]
stroke_followup_paths = [path for path in files if '/Stroke/ses-followup' in path]
stroke_followup_2_paths = [path for path in files if '/Stroke/ses-followup-2' in path]
# stroke_followup2_paths = [path for path in files if '/Stroke/ses-followup2' in path]     # followup2 is after 1 year

glioma_preop_paths = [path for path in files if '/Glioma/ses-preop/' in path]
glioma_postop_paths = [path for path in files if '/Glioma/ses-postop/' in path]
glioma_control_paths = [path for path in files if '/Glioma/ses-control/' in path]

persistences = []
procs = []
threads = 6

#perform_analysis(stroke_acute_paths, 'stroke_acute', True)
#perform_analysis(stroke_control_paths, 'stroke_control', True)
#perform_analysis(stroke_followup_paths, 'stroke_followup', True)
#perform_analysis(stroke_followup_2_paths, 'stroke_followup_2', True)
#perform_analysis(glioma_control_paths, 'glioma_control', True)
#perform_analysis(glioma_postop_paths, 'glioma_postop', True)
#perform_analysis(glioma_preop_paths, 'glioma_preop', True)

#perform_analysis(stroke_acute_paths, 'stroke_acute', False)
#perform_analysis(stroke_control_paths, 'stroke_control', False)
#perform_analysis(stroke_followup_paths, 'stroke_followup', False)
#perform_analysis(stroke_followup_2_paths, 'stroke_followup_2', False)
#perform_analysis(glioma_control_paths, 'glioma_control', False)
perform_analysis(glioma_postop_paths, 'glioma_postop', False)
perform_analysis(glioma_preop_paths, 'glioma_preop', False)

quit()
for subj in stroke_control_paths:
    test(drop_data_in_connect_matrix(load_matrix(subj)), subj.split(os.sep)[-1].split('.')[0])

for subj in stroke_followup_paths:
    test(drop_data_in_connect_matrix(load_matrix(subj)), subj.split(os.sep)[-1].split('.')[0])

for subj in stroke_followup_2_paths:
    test(drop_data_in_connect_matrix(load_matrix(subj)), subj.split(os.sep)[-1].split('.')[0])


for subj in glioma_control_paths:
    test(drop_data_in_connect_matrix(load_matrix(subj)), subj.split(os.sep)[-1].split('.')[0])

for subj in glioma_preop_paths:
    test(drop_data_in_connect_matrix(load_matrix(subj)), subj.split(os.sep)[-1].split('.')[0])

for subj in glioma_postop_paths:
    test(drop_data_in_connect_matrix(load_matrix(subj)), subj.split(os.sep)[-1].split('.')[0])
