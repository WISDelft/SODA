import numpy as np

def normalize_exp(mat):
    new_mat = []
    
    values = []
    for record in mat:
        values.append(record[2])
    mean_v = np.mean(values)
    std_v = np.std(values)
    for record in mat:
        new_record = list(record)
        new_record[2] = float(new_record[2]-mean_v)/std_v
        new_mat.append(new_record)        
    
    return new_mat