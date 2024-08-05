import sys
import pandas as pd
import os
import numpy as np
from glmhmm.glm_hmm import *
from glmhmm.utils import *
sys.path.insert(0, '..')
from preprocessing import utils2 as ut


# Cross Validation
#
# Given a file/folder, number of p.c., whether exclude ignore, and number of folder(default 10)
# return train loglikelihood, test loglikelihood, A, w for all 10 train-test pairs, and train size and test size
#
# The train set of all the folds don't overlap each other
# For each train set, initialize the model twice/three times, and choose the one with the best train loglikelihood
# This is because EM alg doesn't guarantee to find the global minimum
def CrossValidation(path, nprev, exig, num_latent, num_folds = 10, num_init = 3, verbose=False, alldata='../example_data/' ):
    
    # Get data from the given path
    if path is None: #use data from all animals
        data=[]
        for root, dirs, _ in os.walk(alldata):
            for dir in dirs:
                subdir_path = os.path.join(root, dir)
                subdata = ut.Dataloader_ani(subdir_path, nprev = nprev, exig = exig)
                print(subdir_path)
                #print(subdata.shape)
                data.append(subdata)
        data = np.concatenate(data, axis = 0)

    elif isinstance( path, list ): # read data from a list of folders
        data = []
        for subdir in path:
            if os.path.isdir(subdir):
                subdata = ut.Dataloader_ani(subdir, nprev = nprev, exig = exig)
            elif os.path.isdir( alldata+subdir):
                subdata = ut.Dataloader_ani(alldata+subdir, nprev = nprev, exig = exig)
            elif os.path.isfile(subdir):
                subdata = ut.Dataloader_sess(subdir, nprev = nprev, exig = exig)
            elif os.path.isfile(alldata+subdir):
                subdata = ut.Dataloader_sess(alldata+subdir, nprev = nprev, exig = exig)
            data.append(subdata)
        data = np.concatenate(data, axis = 0)
                
    elif os.path.isfile(path):  # single session
        data = ut.Dataloader_sess(path, nprev = nprev, exig = exig)
        
    elif os.path.isdir(path):   # single animal
        data = ut.Dataloader_ani(path, nprev = nprev, exig = exig)
        #print(data.shape)
    
    else:
        raise ValueError("The input is neither a file or a folder")
        
    # Model parameters
    N = len(data) - len(data)//num_folds # number of data/time points
    train_size = N
    test_size = len(data)//num_folds
    K = num_latent # number of latent states
    D = data.shape[1] - 2 # number of GLM inputs (regressors)
    if exig:
        C = 2 # number of observation classes
        prob = "bernoulli"
    else:
        C = 3 # number of observation classes
        prob = "multinomial"
        
    # store values for cross validation
    lls_train = np.zeros((num_init, num_folds))
    lls_test =  np.zeros((num_init, num_folds))
    ll0 =  np.zeros((num_init, num_folds))
    
    A_all = np.zeros((num_init, num_folds,K,K))
    w_all = np.zeros((num_init, num_folds,K,D,C))
    pi0_all = np.zeros((num_init, num_folds,K))

    # Set up the model
    model = GLMHMM(n=N,d=D,c=C,k=K,observations=prob) # set up a new GLM-HMM
    
    # Perform cross-validation
    fold_size = len(data) // num_folds
    for j in range(num_init):
        A_init,w_init,pi_init = model.generate_params() # initialize the model parameters
        for i in range(num_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size
            
            # Split data into train and test sets
            test_data = data[start_idx:end_idx]
            if end_idx < len(data):
                train_data = np.concatenate((data[:start_idx], data[end_idx:]), axis=0)
            else:
                train_data = data[:start_idx]
    
            X_train = train_data[:,1:-1]
            y_train = train_data[:,-1]
            X_test = test_data[:,1:-1]
            y_test = test_data[:,-1]
            probR = np.sum(y_train)/len(y_train)

            model.n = len(y_train)
            ll, A, w, pi0 = model.fit(y_train,X_train,A_init,w_init,pi0=pi_init, fit_init_states=True) # fit the model on trainset
            #print(np.linalg.norm(w-w_init))
            ll = find_last_non_nan_elements(ll.reshape(1, -1))
            lls_train[j, i] = ll[0]
            A_all[j,i] = A
            w_all[j,i] = w
            pi0_all[j,i] = pi0

            # testset
            #GLMHMM.n = test_size
            # convert inferred weights into observation probabilities for each state
            phi = np.zeros((len(X_test),K,C))
            for k in range(K):
                phi[:,k,:] = model.glm.compObs(X_test,w[k,:,:])

            # compute inferred log-likelihoods
            lls_test[j,i],_,_,_ = model.forwardPass(y_test,A,phi)
            ll0[j,i] = np.log(probR) * np.sum(y_test) + np.log(1 - probR) * (len(y_test) - np.sum(y_test)) # base probability
            
        if verbose:
            print('train ll', lls_train[j,i])
            print('test ll',lls_test[j,i])
            print('Init %s complete' %(j+1))
    
    return lls_train, lls_test, ll0, A_all, w_all, pi0_all, train_size, test_size


# Grid Search on number of previous choice and number of latent states
# 
# Given a file/folder, whether exclude the ignore, the largest chosen p.c. and l.s.
# return optimal pc and ls(selected based on test loglikelihood), avg test and train ll,
# std test and train ll (avg and std across folds), train size and test size
# 
# The chosen num of pv are 0,1,2,...,P
# The chosen num of ls are 1,2,...,L
# Consider all the possible combinations(configurations)
# Notice that the returned opt params may not the right one. 
# The better way is to look at the visualization.
def GridSearch(path, exig, P=4, L=7):
    nprevs = np.arange(0,P+1)
    nums_latent = np.arange(1,L+1)
    avg_testll_matrix = np.zeros((P + 1,L))
    avg_trainll_matrix = np.zeros((P+1,L))
    std_testll_matrix = np.zeros((P+1, L))
    std_trainll_matrix = np.zeros((P+1,L))
    for nprev in nprevs:
        for num_latent in nums_latent:
            # lls_train:n, A_all:nxKxK, w_all:nxKxDxC, lls_test:n, where n is the num_fold
            print()
            print("nprev", nprev, "num_latent", num_latent)
            lls_train, lls_test, A_all, w_all, train_size, test_size = CrossValidation(path=path, 
                                                                                       nprev=nprev, exig=exig,
                                                                                       num_latent=num_latent)
            
            avg_testll_matrix[nprev,num_latent-1] = np.mean(lls_test)
            avg_trainll_matrix[nprev,num_latent-1] = np.mean(lls_train)
            std_testll_matrix[nprev,num_latent-1] = np.std(lls_test)
            std_trainll_matrix[nprev,num_latent-1] = np.std(lls_train)

            # Select the optimal configuration using validation/test loglikelihood
            argminx,argminy = np.unravel_index(np.argmax(avg_testll_matrix), avg_testll_matrix.shape)
            opt_nprev = argminx
            opt_latent = argminy + 1
    return opt_nprev, opt_latent, avg_testll_matrix, avg_trainll_matrix, std_testll_matrix, std_trainll_matrix, train_size, test_size



# Fine-tune on small data(a file)
def FineTune(path, nprev, exig, num_latent,A_init,w_init,pi_init=None, tol=3e-4):
    if os.path.isfile(path):
        data = ut.Dataloader_sess(path, nprev = nprev, exig = exig)
    elif os.path.isdir(path):
        data = ut.Dataloader_ani(path, nprev = nprev, exig = exig)
    else:
        raise ValueError("The input is neither a file or a folder")
        
    N = len(data) # number of data/time points
    K = num_latent # number of latent states
    D = data.shape[1] - 2 # number of GLM inputs (regressors)
    if exig:
        C = 2 # number of observation classes
        prob = "bernoulli"
    else:
        C = 3 # number of observation classes
        prob = "multinomial"
    model = GLMHMM(N,D,C,K,observations=prob)
    _,_,_ = model.generate_params()
    X = data[:,1:-1]
    y = data[:,-1]
    ll, A, w, pi0 = model.fit(y,X, A_init,w_init, pi0=pi_init, fit_init_states=True,tol=tol)
    ll = find_last_non_nan_elements(ll.reshape(1, -1))
    y = y.ravel()
    return ll,A,w,pi0, X,y,N,K,D,C


def mostProbSeq( y, A, pi0, phi=None, X=None, w=None ):
    K = A.shape[0]
    N = y.shape[0]
    
    if phi is None:
        if (X is not None) and (w is not None):
            C = w.shape[-1]
            # Emission Probablity matrix
            phi = np.zeros((N,K,C))      
            for i in range(K):
                p = np.exp(X @ w[i,:,:]) 
                phi[:,i,:] = np.divide(p.T,np.sum(p,axis=1)).T
        else:
            raise ValueError("Emission probs cannot be computed")
            
    
    V = np.zeros((K,N))
    Bp = np.zeros((K,N))
    #print(K, N, pi0)
    for s in range(K):
        V[s,0] = np.log(pi0[s] * phi[0, s, int(y[0])]) #pi_s * the prob to get y[0] in state at time 0(phi[0, state, y[0])
        Bp[s,0] = 0
    for t in range(1,N):
        for s in range(K):
            V[s, t] = np.max((V[:, t-1] + np.log(A[:, s])) + np.log(phi[t, s, int(y[t])]))
            Bp[s, t] = np.argmax((V[:, t-1] + np.log(A[:, s])) + np.log(phi[t, s, int(y[t])]))     

    bestpathpointer = np.argmax(V[:,-1])
    bkpointer = bestpathpointer
    bestpath = np.zeros(N)
    for bwt in range(N-1,-1,-1):
        bestpath[bwt] = bkpointer
        bkpointer = Bp[int(bkpointer), bwt]

    return bestpath


def ShuffleControl(path, nprev, exig, num_latent, num_init = 10, num_folds=10, 
                   verbose=False, alldata='../example_data/' ):
    # Get data from the given path
    if path is None: #use data from all animals
        data=[]
        for root, dirs, files in os.walk(alldata):
            for dir in dirs:
                subdir_path = os.path.join(root, dir)
                subdata = ut.Dataloader_ani(subdir_path, nprev = nprev, exig = exig)
                data.append(subdata)
        data = np.concatenate(data, axis = 0)
                
    elif os.path.isfile(path):
        data = ut.Dataloader_sess(path, nprev = nprev, exig = exig)
    elif os.path.isdir(path):
        data = ut.Dataloader_ani(path, nprev = nprev, exig = exig)
        #print(data.shape)
    else:
        raise ValueError("The input is neither a file or a folder")

    N = len(data) - len(data)//num_folds # number of data/time points
    K = num_latent # number of latent states
    D = data.shape[1] - 2 # number of GLM inputs (regressors)
    if exig:
        C = 2 # number of observation classes
        prob = "bernoulli"
    else:
        C = 3 # number of observation classes
        prob = "multinomial"
    
    
    res_orig = { 'lls_train': np.zeros((num_init, num_folds)),    'lls_test': np.zeros((num_init, num_folds)), 'll0':np.zeros((num_init, num_folds)),
                 'A_all': np.zeros((num_init, num_folds,K,K)), 'w_all': np.zeros((num_init, num_folds,K,D,C)), 'pi0_all': np.zeros((num_init, num_folds,K))}
    res_ts = { 'lls_train': np.zeros((num_init, num_folds)),    'lls_test': np.zeros((num_init, num_folds)), 'll0':np.zeros((num_init, num_folds)),
                 'A_all': np.zeros((num_init, num_folds,K,K)), 'w_all': np.zeros((num_init, num_folds,K,D,C)), 'pi0_all': np.zeros((num_init, num_folds,K))}
    res_Xs = { 'lls_train': np.zeros((num_init, num_folds)),    'lls_test': np.zeros((num_init, num_folds)), 'll0':np.zeros((num_init, num_folds)),
                 'A_all': np.zeros((num_init, num_folds,K,K)), 'w_all': np.zeros((num_init, num_folds,K,D,C)), 'pi0_all': np.zeros((num_init, num_folds,K))}

    model = GLMHMM(N,D,C,K,observations=prob)
    fold_size = len(data) // num_folds
    train_size = len(data) - fold_size # same as N
    test_size = fold_size
    alldata = {'orig':{'X':data[:,1:-1], 'y':data[:,-1], 'res':res_orig}, 
                   'shf_t':{'X':data[:,1:-1], 'y':data[:,-1], 'res':res_ts}, 
                   'shf_X':{'X':data[:,1:-1], 'y':data[:,-1], 'res':res_Xs} 
                }
    
    for i in range(num_init):
        A_init,w_init,pi_init = model.generate_params() 

        # Get X and y
        shuffled_indices = np.random.permutation(data.shape[0])
        shfl_time_data = data[shuffled_indices, :] # shuffle the rows of the matrix/shuffle the time
        shfl_X_data = np.hstack((data[shuffled_indices, :-1], data[:, -1:])) # only shuffle the X
        
        X_shf_t = shfl_time_data[:,1:-1]
        y_shf_t = shfl_time_data[:,-1]
        X_shf_X = shfl_X_data[:,1:-1]
        y_shf_X = shfl_X_data[:,-1]

        # update
        alldata['shf_t'].update({'X':X_shf_t, 'y':y_shf_t})
        alldata['shf_X'].update({'X':X_shf_X, 'y':y_shf_X})
                   
        ### Train and test the model
        for j in range(num_folds):
            #print('Fold %s' %j)
            start_idx = i * fold_size
            end_idx = min((i + 1) * fold_size, len(data))
            
            # Split data into train and test sets
            testidx = np.arange(start_idx, end_idx)
            if end_idx < len(data):
                trainidx = np.concatenate((np.arange(0, start_idx), np.arange(end_idx, len(data))), axis=0)
            else:
                trainidx = np.arange(0, start_idx)
            #print('train size', len(trainidx))

            # Fit model for each pair of X and y
            for datatype in ['orig', 'shf_t', 'shf_X']:
                
                X = alldata[datatype]['X']
                y = alldata[datatype]['y']
                res = alldata[datatype]['res']
                probR = np.sum(y[trainidx])/len(trainidx)

                model.n = len(trainidx)
                lls,A,w,pi0 = model.fit(y[trainidx],X[trainidx], A_init,w_init,pi0=pi_init, fit_init_states=True)
                ll = find_last_non_nan_elements(lls.reshape(1, -1))
                res['lls_train'][i,j] = ll
                res['A_all'][i,j] = A
                res['w_all'][i,j] = w
                res['pi0_all'][i,j] = pi0
            
                # convert inferred weights into observation probabilities for each state
                phi = np.zeros((len(X[testidx]),K,C))
                for k in range(K):
                    phi[:,k,:] = model.glm.compObs(X[testidx],w[k,:,:])

                # compute inferred log-likelihoods
                #model.n = len(testidx)
                ll_test,_,_,_ = model.forwardPass(y[testidx],A,phi)
                res['lls_test'][i,j] = ll_test
                res['ll0'][i,j] = np.log(probR) * np.sum(y[trainidx]) + np.log(1 - probR) * (len(y[trainidx]) - np.sum(y[trainidx])) # base probability
                
                ## update resutls
                alldata[datatype].update({'res':res})

    
    return alldata['orig']['res'], alldata['shf_t']['res'], alldata['shf_X']['res'], train_size, test_size

    
