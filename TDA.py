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

plt.rcParams.update({
    "text.usetex": False
})
lineList = [line.rstrip('\n') for line in open('ROIs.txt')]

# Start with the curvature of each patient, 
# then make a distribution for each category and compare across timepoints



def test(matrix, name):
    matrix/= np.max(matrix)
    subj_name = name.split('_')[0]
    os.makedirs(f'Plots/TDA/{subj_name}/', exist_ok=True)

    np.fill_diagonal(matrix, 0)
    Pdmatrix = pd.DataFrame(matrix)
    Pdmatrix.columns = lineList
    Pdmatrix.index = lineList
    Pdmatrix = Pdmatrix.sort_index(0).sort_index(1)
    mask = np.zeros_like(Pdmatrix.values, dtype=np.bool) 
    mask[np.triu_indices_from(mask)] = True 

    # This command will show you the functional connectivity matrix with Seaborn
    plt.figure(figsize = (20, 20))
    _ = sns.heatmap(np.log1p(matrix), cmap='coolwarm', cbar=True, square=False, mask=mask)
    plt.savefig(f'Plots/TDA/{subj_name}/{name}_masked.png')
    plt.close()

    # transform the connectivity matrix in a distance matrix
    matrix = np.ones_like(matrix) * np.max(matrix) - matrix 
    rips_complex = gudhi.RipsComplex(distance_matrix=matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    gudhi.plot_persistence_barcode(diag, legend=True, max_intervals=0)
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
    plt.close()

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

for subj in stroke_acute_paths:
    test(drop_data_in_connect_matrix(load_matrix(subj)), subj.split(os.sep)[-1].split('.')[0])

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
