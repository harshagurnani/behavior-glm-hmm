import numpy as np

import scipy.linalg as ll
import pandas as pd

import ast
from pynwb import NWBHDF5IO, NWBFile, TimeSeries


def data_for_psy( efile , nprev=3 ):
    events = pd.read_csv( efile )
    nTrials = len(events['id'])

    stim = np.zeros( (nTrials,1) )
    choice = np.zeros( nTrials )
    prev_choice = np.zeros((nTrials,nprev))

    stim[events['trial_instruction']=='left']=1
    choice[np.where((events['trial_instruction']=='left') & (events['outcome']=='hit'))[0]] = 1
    choice[np.where((events['trial_instruction']=='right') & (events['outcome']=='miss'))[0]] = 1
    choice[events['outcome']=='ignore']=3

    for k in range(1,nprev+1):
        prev_choice[k:,k-1] = choice[:-k]

    ignore_ids = np.where(choice==3)[0]
    ex_ids = []
    for k in range(0,nprev+1):
        ex_ids.append(k-1)
        for jj in ignore_ids:
            ex_ids.append(jj+k)
    
    use_ids = [jj for jj in range(nTrials) if jj not in ex_ids]
    
    D = dict( y=np.int_(choice[use_ids]), 
             inputs = dict( stim = np.int_(stim[use_ids,:]), 
                           prev_choice = np.int_(prev_choice[use_ids,:]) ) )

    return D


def return_times( events ):
    stim_start = events['sample_start_times'].values # absolute times of sample start
    stim_end = events['sample_stop_times'].values    # abs times for sample stop
    go_start = events['go_start_times'].values       # abs time for go-cue

    #### sometimes stim has multiple values
    for jj in range(len(stim_start)):
        if not isinstance(  stim_start[jj], float ):
            nx1 = ast.literal_eval( stim_start[jj] )
            nx2 = ast.literal_eval( stim_end[jj] )
            if not isinstance( nx1, float):
                nx1=nx1[0]
            if not isinstance( nx2, float):
                nx2=nx2[-1]
            stim_start[jj] = nx1
            stim_end[jj] = nx2
    

    delay = go_start - stim_end     # delay duration in sec
    sample = stim_end - stim_start  # sample duration in sec

    return {'delay_dur':delay, 'sample_dur':sample, 'stim_start':stim_start, 'stim_end':stim_end, 'go_start':go_start }
