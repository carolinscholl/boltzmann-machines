import warnings
warnings.filterwarnings("ignore")

import os
import sys
import env
import tensorflow as tf
import numpy as np
import pickle
from bm.dbm import DBM
from bm.rbm.rbm import BernoulliRBM, logit_mean
from bm.init_BMs import * # helper functions to initialize, fit and load RBMs and 2 layer DBM
from bm.utils.dataset import *
from bm.utils import *
from rbm_utils.stutils import *
from rbm_utils.fimdiag import * # functions to compute the diagonal of the FIM for RBMs
from copy import deepcopy
import argparse
from shutil import copy
import pathlib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from pruning.MNIST_Baselines import *

np.random.seed(42)

# if machine has multiple GPUs only use first one
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_final_pruned_DBM(dbm_path):
    args = get_initial_args()
    args['dbm_dirpath']= dbm_path
    args['rbm1_dirpath'] = None
    args['rbm2_dirpath'] = None
    if os.path.exists(dbm_path):
        dbm = load_dbm_withoutRBMs(Struct(**args))
    else: 
        print("DBM model does not exist")
        dbm = None
    return dbm


def main(dbm_path: str, epochs_retrain: int, pruning_criterion: str):

    print("Retrain DBM and its RBMs saved under path ", dbm_path, f"for {epochs_retrain} epochs each.")

    if not os.path.exists(dbm_path):
        print("DBM does not exist. Specify a valid path.")

    # check that we have access to a GPU and that we only use one!
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        print("Workspace initialized")
    else:
        print("Consider installing GPU version of TF and running sampling from DBMs on GPU.")

    if 'session' in locals() and tf.compat.v1.Session() is not None:
        print('Close interactive session')
        tf.compat.v1.Session().close()

    logreg_digits = get_classifier_trained_on_raw_digits()

    # Load MNIST
    print("\nPreparing data ...\n\n")
    train, test = preprocess_MNIST()
    bin_X_train = train[0]
    y_train = train[1]
    bin_X_test = test[0]
    y_test = test[1]

    n_train = len(bin_X_train)

    dbm = load_dbm_withoutRBMs(Struct(**args))
    params = dbm.get_tf_params(scope='weights')
    W1 = params['W']
    W2 = params['W_1']
    vb = params['vb']
    hb1 = params['hb']
    hb2 = params['hb_1']
    nv=len(vb)
    nh1=len(hb1)
    nh2=len(hb2)
    masks=dbm.get_tf_params(scope='masks')
    prune_mask1 = masks['prune_mask']
    prune_mask2 = masks['prune_mask_1']
    rf_mask1 = masks['rf_mask']
    rf_mask2 = masks['rf_mask_1']
    mask2 = rf_mask2 * prune_mask2
    mask1 = rf_mask1 * prune_mask1

    # set parameters for retraining
    # you can adjust initial parameters by setting the parameters args['w_init'] etc., see get_initial_args() method for parameters
    args['epochs'] = (epochs_retrain, epochs_retrain, epochs_retrain)
    args['n_hidden'] = (nh1, nh2)
    args['lr'] =  (np.logspace(-2,-4,epochs_retrain), np.logspace(-1,-4,epochs_retrain), np.logspace(-1,-4,epochs_retrain), np.logspace(-1,-4,epochs_retrain)) 
    args['prune']=True
    args['n_hidden'] = (nh1, nh2)
    args['freeze_weights']=mask1
    args['random_seed'] = (1337, 1111, 2222, 3333)
    args['v_shape'] = (20,20)
    args['filter_shape']=[(20,20)] # deactivate receptive fields, they are now realised over the prune_mask
    args['n_vis']=nv
    args['rbm1_dirpath'] = os.path.join('models', 'MNIST', 'minimal_models', f'{pruning_criterion}', 'MinimalRBM1')
    args['rbm2_dirpath'] = os.path.join('models', 'MNIST', 'minimal_models', f'{pruning_criterion}', 'MinimalRBM2')
    args['dbm_dirpath'] = os.path.join('models', 'MNIST', 'minimal_models', f'{pruning_criterion}', 'MinimalDBM')

    if not os.path.exists(os.path.join('models', 'MNIST', 'minimal_models', f'{pruning_criterion}')):
        os.makedirs(os.path.join('models', 'MNIST', 'minimal_models', f'{pruning_criterion}'))

    print('Retrain RBM1')
    rbm1 = make_rbm1(bin_X_train, Struct(**args))

    #run on gpu
    config = tf.ConfigProto(
            device_count = {'GPU': 1})
    rbm1._tf_session_config = config
    Q_train = rbm1.transform(bin_X_train) 
    Q_test = rbm1.transform(bin_X_test) 

    Q_train_bin = make_probs_binary(Q_train)
    Q_test_bin = make_probs_binary(Q_test)

    print('Retrain RBM2')
    args['freeze_weights'] = mask2
    rbm2 = make_rbm2(Q_train_bin, Struct(**args))

    G_train = rbm2.transform(Q_train_bin)
    G_test = rbm2.transform(Q_test_bin)

    G_train_bin = make_probs_binary(G_train)
    G_test_bin = make_probs_binary(G_test)

    print('Retrain DBM')
    dbm = make_dbm(bin_X_train, None, (rbm1, rbm2), Q_train_bin, G_train_bin, Struct(**args))

    print('Sampling from retrained DBM')

    SAMPLE_EVERY = 200
    #run on gpu
    config = tf.ConfigProto(
            device_count = {'GPU': 1})
    dbm._tf_session_config = config
    samples = dbm.sample_gibbs(n_gibbs_steps=SAMPLE_EVERY, save_model=False, n_runs=n_train)
    s_v = samples[:,:nv]
    s_h1 = samples[:,nv:nv+nh1]
    s_h2 = samples[:,nv+nh1:]

    print("\nEvaluate samples similarity to digits by Logistic regression...")
    pred_probs_samples_logreg = logreg_digits.predict_proba(s_v)
    prob_winner_pred_samples = pred_probs_samples_logreg.max(axis=1)
    mean_qual = np.mean(prob_winner_pred_samples)
    print("Mean quality of samples", mean_qual, "Std: ", np.std(prob_winner_pred_samples))

    ind_winner_pred_samples = np.argmax(pred_probs_samples_logreg, axis=1)
    sample_class, sample_counts = np.unique(ind_winner_pred_samples, return_counts=True)
    dic = dict(zip(sample_class, sample_counts))
    print("sample counts per class",dic)

    mean_qual_d = np.zeros(len(sample_class))

    for i in range(0,len(sample_class)):
        ind=np.where(ind_winner_pred_samples==sample_class[i])
        mean_qual_d[i] = np.mean(prob_winner_pred_samples[ind])

    qual_d = [sample_class, mean_qual_d, sample_counts]

    print("Quality per digit class of Logistic regression for samples")
    qual_d= np.asarray(qual_d)
    for i in range(len(qual_d.T)):
        print("digit", qual_d[0,i], "confidence", qual_d[1,i], "counts", qual_d[2,i])

    np.save(os.path.join('models', 'MNIST', 'minimal_models', f'{pruning_criterion}', 'fi_minimal_model_qualdigits.npy'), qual_d)


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
                              'FI_DIAG': 'Prune least important weights according to FIM diagonal', 
                              'HEURISTIC_DIAG': 'Prune least important weights according to heuristic FI estimate'}
        if not value in valid_pruning_criteria.keys():
            print('Choose a valid pruning criterion: ')
            for key,value in valid_pruning_criteria.items():
                print(key, ":", value)
            raise argparse.ArgumentTypeError('Invalid pruning criterion')
        return value

    parser = argparse.ArgumentParser(description = 'DBM Retraining')
    parser.add_argument('dbm_path', help='Path to DBM to be retrained', type=str)
    parser.add_argument('pruning_criterion', help='Pruning criterion', type=check_validity_pruning_criterion)
    parser.add_argument('n_epochs_retrain', default=20, nargs='?', help='Number of epochs for retraining', type=check_positive)

    args = parser.parse_args()

    main(args.dbm_path, args.pruning_criterion, args.n_epochs_retrain)
