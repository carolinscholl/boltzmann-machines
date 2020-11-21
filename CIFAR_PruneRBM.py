import warnings
warnings.filterwarnings("ignore")

import os
import sys
import tensorflow as tf
import numpy as np
import pickle
from bm.rbm.rbm import BernoulliRBM, logit_mean
from bm.utils.dataset import *
from bm.init_BMs import * # helper functions to initialize, fit and load RBMs and 2 layer DBM
from rbm_utils.stutils import *
from rbm_utils.FIMDiag import * # functions to compute the diagonal of the FIM for RBMs
from copy import deepcopy
import argparse

np.random.seed(42)

# if machine has multiple GPUs only use first one
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def save_res(results_path, params=None, indices_hiddens=None, samples=None, mask=None,fi=None):
    '''
    save the results (parameters, masks and samples) after pruning.
    parameters are expected to be a dictionary, samples are the binary samples from the model.
    '''
    if params is not None:
        f = open(results_path+'params.pkl',"wb")
        pickle.dump(params,f)
        f.close()

    if fi is not None:
        np.save(results_path+'fi.npy', fi)

    if indices_hiddens is not None:
        np.save(results_path+'indices_kept_hiddens.npy', indices_hiddens)

    if mask is not None: # only need to save mask once per iteration
        np.save(results_path+'mask.npy', np.array(mask).astype(np.bool))

    if samples is not None:
        np.save(results_path+'samples.npy', np.array(samples).astype(np.bool))

def main(pruning_criterion, percentile=50, n_hidden=70, n_pruning_session=3):

    # check that we have access to a GPU 
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        print("Workspace initialized")
    else:
        print("Consider installing GPU version of TF and running sampling from RBMs on GPU.")

    if sys.version_info[0] == 2:
        def unpickle(fo):
            import cPickle
            dict = cPickle.load(fo)
            fo.close()
            return dict

    # load the data
    radius = 2.3
    n_train = 90000 
    n_val = n_train

    # check that image data is available
    try:
        data = load_cifar_circles(os.path.join('data', 'cifar-10-batches-py/'), radius)
        pass
    except(FileNotFoundError, IOError):
        print("Cannot find CIFAR image data, please run data/fetch_cifar10.sh first")
        return
        
    X_train = data[np.random.permutation(data.shape[0])[:n_train],:]
    X_val = data[np.random.permutation(data.shape[0])[:n_val],:]

    nv = X_train.shape[1] # visible layer size = pixels in radius of CIFAR circles
    nh = n_hidden # number of hidden units
    session = 0 # before pruning
    n_sessions = n_pruning_session # number of pruning sessions

    # create results paths
    top_folder = os.path.join('models', 'CIFAR','{}v'.format(nv)+'{}h'.format(nh))
    if not os.path.exists(top_folder):
        os.makedirs(top_folder)
    
    model_path = os.path.join(top_folder, pruning_criterion+'{}v'.format(nv)+'{}h'.format(nh))
    assert not os.path.exists(model_path), "model path already exists - abort"
    os.mkdir(model_path)

    compare_path = os.path.join(top_folder, 'Initial{}v{}h/'.format(nv,nh))
    if os.path.exists(compare_path):
        initial_exists= True
    else:
        initial_exists = False # if true, we already have an existing initial RBM that we need to use to initialize the others

    if not initial_exists:
        os.mkdir(compare_path)
        results_path = os.path.join(compare_path, 'results')
        os.mkdir(results_path) # results path

        args = {}
        args['epochs'] = 2
        args['n_hidden'] = nh 
        args['n_visible'] = nv
        args['prune']=True
        args['filter_shape']=None
        args['freeze_weights']=None # will be initialized with mask of ones
        args['w_init'] = 0.1*1 
        args['vb_init'] = logit_mean(X_train) 
        args['hb_init'] = -2.0
        args['n_gibbs_steps'] = 1 
        args['lr'] = 1*np.logspace(-1,-2,args['epochs']) 
        args['max_epoch'] = args['epochs']
        args['batch_size'] = 1
        args['l2'] = 0 
        args['momentum'] = 0.9
        args['sample_v_states'] = True
        args['dropout'] = None
        args['sparsity_target'] = 0.1
        args['sparsity_cost'] = 0 
        args['sparsity_damping'] = 0.9
        args['random_seed'] = 666
        args['dtype'] = 'float32'
        args['model_dirpath'] = os.path.join(compare_path,'{}v{}h_session{}/'.format(nv, nh, session))
        print('learning rates: ',args['lr'])

        # initialize RBM and save initial parameter
        rbm = init_rbm(Struct(**args))

        w_before = rbm.get_tf_params(scope='weights')
        m = rbm.get_tf_params(scope='masks')
        p_mask = m['prune_mask']
        s = rbm.sample_gibbs(n_gibbs_steps=200, save_model=False, n_runs=n_train)

        var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w_before['W'])

        cur_res_path = os.path.join(results_path, 'session{}_untrained_'.format(session))
        save_res(cur_res_path, params=w_before, samples=s, mask=p_mask ,fi=var_est)

        # train the intiial RBM
        rbm.fit(X_train, X_val)

        w_after = rbm.get_tf_params(scope='weights')
        config = tf.ConfigProto(device_count = {'GPU': 1})
        rbm._tf_session_config = config
        s = rbm.sample_gibbs(n_gibbs_steps=200, save_model=False, n_runs=n_train)

        var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w_after['W'])

        cur_res_path = os.path.join(results_path, 'session{}_trained_'.format(session))
        save_res(cur_res_path, params=w_after, samples=s, fi=var_est)


    # now change the results path
    results_path = os.path.join(model_path, 'results')
    os.mkdir(results_path) # results path

    print('load initial trained RBM parameters')

    # load initial parameters
    infile = open(os.path.join(compare_path, 'results', 'session0_trained_params.pkl'),'rb')
    w_after = pickle.load(infile)
    infile.close()

    cur_res_path = os.path.join(results_path, 'session{}_trained_before_init'.format(session))
    save_res(cur_res_path, w_after)

    # assign the trained parameters to the RBM which will be pruned according to set criterion
    args = {}
    args['epochs'] = 2
    args['n_hidden'] = nh 
    args['n_visible'] = nv
    args['prune']=True
    args['freeze_weights']=None 
    args['filter_shape']=None
    args['w_init'] = w_after['W']
    args['vb_init'] = w_after['vb']
    args['hb_init'] = w_after['hb']
    args['n_gibbs_steps'] = 1 
    args['lr'] = 1*np.logspace(-1,-2,args['epochs']) 
    args['max_epoch'] = args['epochs']
    args['batch_size'] = 1
    args['l2'] = 0 
    args['momentum'] = 0.9
    args['sample_v_states'] = True
    args['dropout'] = None
    args['sparsity_target'] = 0.1
    args['sparsity_cost'] = 0 #0.01
    args['sparsity_damping'] = 0.9
    args['random_seed'] = 666
    args['dtype'] = 'float32'
    args['model_dirpath'] = os.path.join(model_path,'{}v{}h_session{}/'.format(nv, nh, session))
    print('learning rates: ',args['lr'])

    # initialize RBM and save initial parameter
    rbm = init_rbm(Struct(**args))

    # now that it has the same parameters as the RBM we compare it to, we would not have to save anything,
    # let's do it however, so that we can verify that the parameters are indeed the same and the samples are similar
    w_from_model = rbm.get_tf_params(scope='weights')
    assert np.all(w_from_model['W']==w_after['W']), "weights changed during initialization, abort"
    w_after = deepcopy(w_from_model)
    config = tf.ConfigProto(device_count = {'GPU': 1})
    rbm._tf_session_config = config
    s = rbm.sample_gibbs(n_gibbs_steps=200, save_model=False, n_runs=n_train)
    m = rbm.get_tf_params(scope='masks')
    p_mask = m['prune_mask']
    cur_res_path = os.path.join(results_path, 'session{}_trained_'.format(session))

    var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w_after['W'])

    save_res(cur_res_path, params=w_after, samples=s, fi=var_est)

    if pruning_criterion == 'ANTI':
        # if anti-FI is chosen, we remove the most important ones
        percentile = int(100-percentile)

    # start pruning sessions
    for sess in range(n_sessions):

        # copy the last parameters after last training
        w = w_after['W']
        vb = w_after['vb']
        hb = w_after['hb']

        temp_mask = p_mask

        # prune
        if pruning_criterion == 'FI_DIAG':
            print('Weight pruning according to diagonal values of the FIM for each weight')

            if sess >0: # in session 0 we take the fi computed above
                var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w)

            fi_weights = var_est.reshape((nh,nv)).T * temp_mask

            fim_diag = deepcopy(fi_weights)

            perc = np.percentile(fi_weights[np.where(temp_mask!=0)],percentile)

            print(sum(fi_weights[np.where(temp_mask!=0)].flatten()<=perc), "weights of a total of",
            len(w[np.where(temp_mask!=0)].flatten()), "are pruned: ",
            sum(fi_weights[np.where(temp_mask!=0)].flatten()<=perc) /
                len(w[np.where(temp_mask!=0)].flatten()), "of all weights.")
            print("Weights with FI lower than", perc, "pruned")

            keep = np.reshape(fi_weights, (nv, nh)) > perc

        elif pruning_criterion == 'FI_DIAG_SQUARED':
            print('Weight pruning according to squared diagonal values of the FIM for each weight')

            if sess >0: # in session 0 we take the fi computed above
                var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w)

            fim_diag = var_est.reshape((nh,nv)).T * temp_mask

            fi_weights = np.square(var_est)
            fi_weights = var_est.reshape((nh,nv)).T * temp_mask

            perc = np.percentile(fi_weights[np.where(temp_mask!=0)],percentile)

            print(sum(fi_weights[np.where(temp_mask!=0)].flatten()<=perc), "weights of a total of",
            len(w[np.where(temp_mask!=0)].flatten()), "are pruned: ",
            sum(fi_weights[np.where(temp_mask!=0)].flatten()<=perc) /
                len(w[np.where(temp_mask!=0)].flatten()), "of all weights.")
            print("Weights with FI lower than", perc, "pruned")

            keep = np.reshape(fi_weights, (nv, nh)) > perc

        elif pruning_criterion =='HEURISTIC_DIAG':
            # heuristic estimate of FI diagonal
            print('Weight pruning according to heuristic estimates of the FIM for each weight')

            if sess >0: # in session 0 we take the fi computed above
                var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w)

            fim_diag = var_est.reshape((nh,nv)).T * temp_mask # we save this each time

            fi_weights = heu_est.reshape((nh,nv)).T * temp_mask # we use this for pruning

            perc = np.percentile(fi_weights[np.where(temp_mask!=0)],percentile)

            print(sum(fi_weights[np.where(temp_mask!=0)].flatten()<=perc), "weights of a total of",
            len(w[np.where(temp_mask!=0)].flatten()), "are pruned: ",
            sum(fi_weights[np.where(temp_mask!=0)].flatten()<=perc) /
                len(w[np.where(temp_mask!=0)].flatten()), "of all weights.")
            print("Weights with FI lower than", perc, "pruned")

            keep = np.reshape(fi_weights, (nv, nh)) > perc

        elif pruning_criterion == 'FIM_EIGENVECTOR':
            # prune according to weight-specific entry of first eigenvector
            print('Weight pruning according to first eigenvector of FIM')
            fim = rbm_fim_numpy(s.astype(bool), nv) # nv = number of visible units
            fim_eigvals, vec = fim_eig(fim, nv, return_eigenvectors=True)
            fim_visbias, fim_hidbias, fim_weights = vec

            fi_weights = fim_weights[:,:,0] # take first eigenvector
            fi_weights = np.reshape(np.array(fi_weights), (nv,nh)) * temp_mask

            if sess >0: # in session 0 we take the fi computed above
                var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w)

            fim_diag = var_est.reshape((nh,nv)).T * temp_mask # we save this each time

            perc = np.percentile(fi_weights[np.where(temp_mask!=0)],percentile)

            print(sum(fi_weights[np.where(temp_mask!=0)].flatten()<=perc), "weights of a total of",
            len(w[np.where(temp_mask!=0)].flatten()), "are pruned: ",
            sum(fi_weights[np.where(temp_mask!=0)].flatten()<=perc) /
                len(w[np.where(temp_mask!=0)].flatten()), "of all weights.")
            print("Weights with FI lower than", perc, "pruned")

            keep = np.reshape(fi_weights, (nv, nh)) > perc

        elif pruning_criterion == 'WEIGHTMAG':
            # prune according to absolute weight magnitude
            print('Weight pruning according to absolute weight magnitude')
            if sess >0: # in session 0 we take the fi computed above
                var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w)

            fim_diag = var_est.reshape((nh,nv)).T * temp_mask # we save this each time

            weights = deepcopy(w)

            weights = np.abs(weights).reshape((nv,nh)) * temp_mask

            perc = np.percentile(weights[np.where(temp_mask!=0)],percentile)

            print(sum(weights[np.where(temp_mask!=0)].flatten()<=perc), "weights of a total of",
            len(w[np.where(temp_mask!=0)].flatten()), "are pruned: ",
            sum(weights[np.where(temp_mask!=0)].flatten()<=perc) /
                len(w[np.where(temp_mask!=0)].flatten()), "of all weights.")
            print("Weights with FI lower than", perc, "pruned")

            keep = np.reshape(weights, (nv, nh)) > perc

        elif pruning_criterion == 'ANTI':

            print("Prune", percentile, " percent of weights with highest FI (anti-FI pruning)")
            if sess >0: # in session 0 we take the fi computed above
                var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w)

            fi_weights = var_est.reshape((nh,nv)).T * temp_mask
            fim_diag = deepcopy(fi_weights)

            perc = np.percentile(fi_weights[np.where(temp_mask!=0)],percentile)

            n_pruned = sum(fi_weights[np.where(temp_mask!=0)].flatten()>perc)
            print(n_pruned, "weights of a total of",
                len(w[np.where(temp_mask!=0)].flatten()), "are pruned: ",
                n_pruned/len(w[np.where(temp_mask!=0)].flatten()), "of all weights.")
            print("Weights with FI higher than", perc, "pruned")

            keep = np.reshape(fi_weights, (nv, nh)) <= perc

        elif pruning_criterion == 'RANDOM':
            print("Randomly prune ", percentile, " percent of weights.")

            if sess >0: # in session 0 we take the fi computed above
                var_est, heu_est = FI_weights_var_heur_estimates(s, nv, nh, w)

            fi_weights = var_est.reshape((nh,nv)).T * temp_mask
            fim_diag = deepcopy(fi_weights) # we save this to look at it over time

            indices_that_still_exist = np.where(temp_mask.flatten().astype(bool))
            indices_that_still_exist = np.squeeze(np.asarray(indices_that_still_exist))

            # that many we have to remove in order to delete percentile
            n_to_remove = int((percentile/100)*sum(temp_mask.flatten()!=0).flatten())

            # randomly select n weights
            selected = np.random.choice(indices_that_still_exist, n_to_remove, replace=False)

            keep = temp_mask.flatten()
            keep[selected] = False # set them to false
            keep = keep.reshape(nv, nh)

        keep[temp_mask==0]=0 # these weights don't exist anyways

        # check how many hidden units in first layer are still connected
        no_left_hidden = 0 # number of hidden units left
        indices_of_left_hiddens = [] # indices of the ones we keep
        for j in range(nh):
            if sum(keep[:,j]!=0):
                no_left_hidden+=1
                indices_of_left_hiddens.append(j)
        print(no_left_hidden, "hidden units in first layer are still connected by weights to visibles.")

        active_nh = no_left_hidden

        no_left_visible = 0
        indices_of_left_visibles = []
        indices_of_lost_visibles = []
        for j in range(nv):
            if sum(keep[j]!=0):
                no_left_visible+=1
                indices_of_left_visibles.append(j)
            else:
                indices_of_lost_visibles.append(j)
        print(no_left_visible, "visible units are still connected by weights.", nv-no_left_visible, "unconnected visible units.")

        print("Indices of lost visibles:", indices_of_lost_visibles)

        new_weights = deepcopy(w)

        # remove unconnected hidden units
        new_weights = new_weights[:, indices_of_left_hiddens]
        keep = keep[:, indices_of_left_hiddens]
        new_weights[keep==0]=0
        new_hb = hb[indices_of_left_hiddens]

        args['model_dirpath'] = os.path.join(model_path,'{}v{}h_session{}/'.format(nv,nh,sess+1))
        args['vb_init']= vb
        args['hb_init']= new_hb
        args['prune']=True
        args['n_hidden'] = active_nh
        args['freeze_weights']=keep
        args['w_init']= new_weights
        args['n_vis']=nv
        args['n_hidden'] = active_nh

        rbm_pruned = init_rbm(Struct(**args))

        w_before = rbm_pruned.get_tf_params(scope='weights')
        m = rbm_pruned.get_tf_params(scope='masks')
        p_mask = m['prune_mask']

        config = tf.ConfigProto(
            device_count = {'GPU': 1})
        rbm_pruned._tf_session_config = config

        s = rbm_pruned.sample_gibbs(n_gibbs_steps=200, save_model=False, n_runs=n_train)

        cur_res_path = os.path.join(results_path, 'session{}_untrained_'.format(sess+1))
        save_res(cur_res_path, params=w_before, indices_hiddens=np.array(indices_of_left_hiddens), samples=s, mask=p_mask, fi=fim_diag)

        # retrain
        rbm_pruned.fit(X_train, X_val)

        w_after = rbm_pruned.get_tf_params(scope='weights')
        config = tf.ConfigProto(device_count = {'GPU': 1})
        rbm_pruned._tf_session_config = config
        s = rbm_pruned.sample_gibbs(n_gibbs_steps=200, save_model=False, n_runs=n_train)

        var_est, heu_est = FI_weights_var_heur_estimates(s, nv, active_nh, w)

        fi_weights = var_est.reshape((active_nh,nv)).T * p_mask

        cur_res_path = os.path.join(results_path, 'session{}_trained_'.format(sess+1))
        save_res(cur_res_path, params=w_after, samples=s, fi=fi_weights)

        nh = active_nh



if __name__ == '__main__':

    def check_positive(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError('Not a positive integer.')
        return ivalue
    
    def check_validity_pruning_criterion(value):
        valid_pruning_criteria = {'ANTI': 'Prune most important weights according to FIM diagonal',
                              'RANDOM': 'Randomly prune weights',
                              'WEIGHTMAG': 'Prune weights with smallest magnitude', 
                              'FIM_EIGENVECTOR': 'Prune least important weights according to first eigenvector of FIM',
                              'FI_DIAG': 'Prune least important weights according to FIM diagonal', 
                              'FI_DIAG_SQUARED': 'Prune least important weights according to square of FIM diagonal',
                              'HEURISTIC_DIAG': 'Prune least important weights according to heuristic FI estimate'}
        if not value in valid_pruning_criteria.keys():
            print('Choose a valid pruning criterion: ')
            for key,value in valid_pruning_criteria.items():
                print(key, ":", value)
            raise argparse.ArgumentTypeError('Invalid pruning criterion')
        return value

    parser = argparse.ArgumentParser(description = 'RBM Pruning')
    parser.add_argument('pruning_criterion', help='Pruning criterion', type=check_validity_pruning_criterion)
    parser.add_argument('percentile', default=50, nargs='?', help='Percentage of weights removed in each iteration', type=int, choices=range(1, 100))
    parser.add_argument('n_hidden', default=70, nargs='?', help='Number of hidden units', type=check_positive)
    parser.add_argument('n_pruning_session', default=3, nargs='?', help='Number of pruning sessions', type=check_positive)

    args = parser.parse_args()

    main(args.pruning_criterion, args.percentile, args.n_hidden, args.n_pruning_session)