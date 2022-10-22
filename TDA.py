import networkx as nx
import nxviz
import gudhi
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pandas as pd

from utils.paths import get_subjects
from TDAlib import *
import os

lineList = [line.rstrip('\n') for line in open('ROIs.txt')]

# Start with the curvature of each patient, 
# then make a distribution for each category and compare across timepoints



def test(matrix, name):
    #print(graph)
    rips_complex = gudhi.RipsComplex(distance_matrix=matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    gudhi.plot_persistence_barcode(diag, legend=True, max_intervals=0)
    plt.savefig(f'Plots/{name}_persistence_barcode.png')
    plt.close()

    gudhi.plot_persistence_diagram(diag, legend=True, max_intervals=0)
    plt.tick_params(axis='both', labelsize=15)
    plt.xlabel('Birth', fontsize=15)
    plt.ylabel('Death', fontsize=15)
    plt.savefig(f'Plots/{name}_persistence_diagram.png')
    plt.close()

    gudhi.plot_persistence_density(diag, dimension=1)
    plt.savefig(f'Plots/{name}_persistence_density.png')
    plt.close()

    plotEuler_thr(matrix, 70)
    plt.savefig(f'Plots/{name}_euler_entrophy.png')

    curvalues = Curv_thr(matrix, e=0.7)
    dict(zip(lineList[:10], curvalues[:10]))
    plt.figure(figsize=(20,5))
    sns.distplot(curvalues, kde=True, norm_hist=False)
    plt.xlabel('Curvature values')
    plt.ylabel('Counts')
    plt.savefig(f'Plots/{name}_curvature.png')
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

stroke_acute_paths = [path for path in files if '/Stroke/ses-acute/' in path]
stroke_control_paths = [path for path in files if '/Stroke/ses-control' in path]
stroke_followup_paths = [path for path in files if '/Stroke/ses-followup' in path]
stroke_followup_2_paths = [path for path in files if '/Stroke/ses-followup-2' in path]
# stroke_followup2_paths = [path for path in files if '/Stroke/ses-followup2' in path]     # followup2 is after 1 year

glioma_preop_paths = [path for path in files if '/Glioma/ses-preop/' in path]
glioma_postop_paths = [path for path in files if '/Glioma/ses-postop/' in path]
glioma_control_paths = [path for path in files if '/Glioma/ses-control/' in path]

for s_a in stroke_acute_paths:
    test(drop_data_in_connect_matrix(load_matrix(s_a)), s_a.split(os.sep)[-1].split('.')[0])
