import sys
import pandas as pd
import os
import numpy as np

# I add a new argument exclude_ignore.
def data_for_psy(efile, nprev=3, exclude_ignore = True):
    events = pd.read_csv( efile )
    nTrials = len(events['id'])
    
    stim = np.zeros( (nTrials,1) )
    choice = np.zeros( nTrials )
    prev_choice = np.zeros((nTrials,nprev))
    prev_stim = np.zeros((nTrials, 1))
    
    stim[events['trial_instruction']=='left']=1

    # right: 0, left:1, ignore:2
    choice[np.where((events['trial_instruction']=='left') & (events['outcome']=='hit'))[0]] = 1
    choice[np.where((events['trial_instruction']=='right') & (events['outcome']=='miss'))[0]] = 1
    choice[events['outcome']=='ignore']=2

    prev_stim[1:] = stim[:-1]
    for k in range(1,nprev+1):
        prev_choice[k:,k-1] = choice[:-k]
    
    ignore_ids = np.where(choice==2)[0]
    ex_ids = []
    for k in range(0,nprev+1):
        ex_ids.append(k-1)

        # if the ignore is included, then don't exclude the following ids
        if (exclude_ignore):
            for jj in ignore_ids:
                ex_ids.append(jj+k)
            
    use_ids = [jj for jj in range(nTrials) if jj not in ex_ids]

    D = {
        'y': np.int_(choice[use_ids]), 
        'inputs': {
            'stim': np.int_(stim[use_ids, :]), 
            'prev_choice': np.int_(prev_choice[use_ids, :]), 
            'prev_stim': np.int_(prev_stim[use_ids, :])  
        }
    }
    return D, use_ids


def Dataloader_sess(file, nprev=3, exig=True):

    D, use_id = data_for_psy(file, nprev = nprev, exclude_ignore = exig)
    
    # t = np.array(use_id).reshape(-1, 1)
    t = np.array(range(1, len(D['y']) + 1)).reshape(-1,1)
    x = np.concatenate([np.ones((len(D['y']), 1)), D['inputs']['stim'], D['inputs']['prev_choice'], D['inputs']['prev_stim']], axis = 1)
    y = D['y'].reshape(-1,1)

    # matrix = t| x |y
    matrix = np.concatenate([t,x,y], axis = 1)
    sess = np.array([0, len(matrix)])
    

    return matrix, sess


def Dataloader_ani(directory, nprev = 3, exig = True):
    # Loop through all files in the directory
    sess = [0]
    matrix_list = []
    accum_t = 0
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            submatrix, subsess = Dataloader_sess(directory + '/' +  filename, nprev = nprev, exig = exig)
            accum_t += len(submatrix)
            matrix_list.append(submatrix)
            sess.append(accum_t)
    matrix = np.concatenate(matrix_list, axis = 0)
    matrix[:,0] = np.arange(1, len(matrix) + 1)
    sess = np.array(sess)
    return matrix, sess