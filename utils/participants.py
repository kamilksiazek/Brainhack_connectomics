import pandas as pd
import numpy as np

def load_tumor_info(subject_list):
    ######################
    ### Tumor Features ###
    ######################
    info = pd.read_csv('datasets/participants.tsv', sep='\t')
    info = info[info["participant_id"].str.contains("CON") == False]
    info.set_index(info.participant_id, inplace=True)
    info.drop(['participant_id'], axis=1, inplace=True)
    info.index.name = None

    tumor_sizes = np.array([dict(info["tumor size (cub cm)"])[k] for k in subject_list])
    tumor_types = np.array([1 if 'ningioma' in dict(info["tumor type & grade"])[k] else 2 for k in subject_list])
    tumor_locs = np.array([1 if 'Frontal' in dict(info["tumor location"])[k] else 2 for k in subject_list])
    tumor_grade = np.array([2 if 'II' in dict(info["tumor type & grade"])[k] else 1 for k in subject_list])
    tumor_ventricles = np.array([2 if 'yes' in dict(info["ventricles"])[k] else 1 for k in subject_list])
    
    return tumor_sizes, tumor_types, tumor_locs, tumor_grade, tumor_ventricles

def load_stroke_info(subject_list):
    #######################
    ### Stroke Features ###
    #######################
    pass

if __name__ == '__main__':
    pass