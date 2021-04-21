import warnings
warnings.filterwarnings("ignore")

import os
import sys
import env
import tensorflow as tf
import numpy as np
import pickle
import random
from bm.dbm import DBM
from bm.rbm.rbm import BernoulliRBM, logit_mean
from bm.init_BMs import * # helper functions to initialize, fit and load RBMs and 2 layer DBM
from bm.utils.dataset import *
from bm.utils import *
from bm.utils.plot_utils import im_plot
from rbm_utils.stutils import *
from rbm_utils.fimdiag import * # functions to compute the diagonal of the FIM for RBMs
from copy import deepcopy
from shutil import copy
import argparse
import pathlib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

# if machine has multiple GPUs only use first one
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def _crop_MNIST(X):
    X_20 = []
    for img in X:
        img_re = np.reshape(img, (28, 28))
        crop_img = img_re[4:24, 4:24]
        X_20.append(crop_img.flatten())  
    return X_20

def _binarize_by_mean(X):
    imgmean_total = np.mean(X) 
    bin_X_train = ((np.array([np.sign(d - imgmean_total) for d in X]) + 1) / 2).astype(bool)
    return bin_X_train

def preprocess_MNIST(image_path=None):
    if image_path is None: 
        image_path = os.path.join('..', 'data')

    # check that image data is available
    try:
        X_train, y_train = load_mnist(mode='train', path=image_path)
        X_test, y_test = load_mnist(mode='test', path=image_path)
        pass
    except(FileNotFoundError, IOError):
        print("Cannot find MNIST image data, please run data/fetch_mnist.sh first")
        raise

    X_train /= 255.
    X_test /= 255.

    RNG(seed=42).shuffle(X_train)
    RNG(seed=42).shuffle(y_train)

    # crop and binarize
    X_train = _crop_MNIST(X_train)
    X_test = _crop_MNIST(X_test)

    bin_X_train = _binarize_by_mean(X_train)
    bin_X_test = _binarize_by_mean(X_test)

    # distribution/balance of digit classes
    #dig_class, dig_counts = np.unique(y_train, return_counts=True)
    #dig = dict(zip(dig_class, dig_counts))
    #print('Class distribution of training data', dig)
    
    return (bin_X_train, y_train), (bin_X_test, y_test)

def get_classifier_trained_on_raw_digits(path=None):
    # retrieve logistic regression classifier trained on 20x20 binary MNIST
    if path is None: 
        path = os.path.join('..', 'models', 'MNIST', 'logreg_MNIST.pkl')
    
    if not os.path.exists(path):
        logreg_digits = None  # avoid retraining of classifier trained on raw digits
        print('logistic regression classifier trained on raw digits not found')
        # logreg_digits = train_classifier_on_raw_digits(path)
    else:
        logreg_digits = joblib.load(path) 

    return logreg_digits 
    
def evaluate_classifier_trained_on_raw_digits_on_random_patterns(save_path, logreg_digits):

    random_patterns = np.random.choice([0, 1], size=[60000,400])
        
    #print("\nEvaluate random pattern's similarity to digits by Logistic regression...")
    pred_probs_random_logreg = logreg_digits.predict_proba(random_patterns)
    prob_winner_pred_random = pred_probs_random_logreg.max(axis=1)
    mean_qual = np.mean(prob_winner_pred_random)
    #print("Mean quality of random patterns", mean_qual, "Std: ", np.std(prob_winner_pred_random))

    ind_winner_pred_random = np.argmax(pred_probs_random_logreg, axis=1)
    random_class, random_counts = np.unique(ind_winner_pred_random, return_counts=True)
    dic = dict(zip(random_class, random_counts))
    #print("sample counts per class",dic)

    mean_qual_d = np.zeros(len(random_class))

    for i in range(0,len(random_class)):
        ind=np.where(ind_winner_pred_random==random_class[i])
        mean_qual_d[i] = np.mean(prob_winner_pred_random[ind])

    qual_d_random = [random_class, mean_qual_d, random_counts]
    np.save(os.path.join(save_path, 'ProbsWinDig_Random.npy'), qual_d_random)

def evaluate_classifier_trained_on_raw_digits_on_generated_samples(s_v, logreg, DBM_path):
    # s_v = visible layer samples of size 20x20

    logreg_digits = logreg

    #print("\nEvaluate samples similarity to digits by Logistic regression...")
    pred_probs_samples_logreg = logreg_digits.predict_proba(s_v)
    prob_winner_pred_samples = pred_probs_samples_logreg.max(axis=1)
    mean_qual = np.mean(prob_winner_pred_samples)
    #print("Mean quality of samples", mean_qual, "Std: ", np.std(prob_winner_pred_samples))

    ind_winner_pred_samples = np.argmax(pred_probs_samples_logreg, axis=1)
    sample_class, sample_counts = np.unique(ind_winner_pred_samples, return_counts=True)
    dic = dict(zip(sample_class, sample_counts))
    #print("sample counts per class",dic)

    mean_qual_d = np.zeros(len(sample_class))

    for i in range(0,len(sample_class)):
        ind=np.where(ind_winner_pred_samples==sample_class[i])
        mean_qual_d[i] = np.mean(prob_winner_pred_samples[ind])

    qual_d = [sample_class, mean_qual_d, sample_counts]
    np.save(os.path.join(DBM_path, 'ProbsWinDig_Initial_Samples.npy'), qual_d)

def evaluate_classifier_trained_on_raw_digits_on_testdigits(save_path, logreg_digits):
    # get training data
    _, test = preprocess_MNIST()

    bin_X_test = test[0]
    y_test = test[1]

    acc = logreg_digits.score(bin_X_test, y_test)
    np.save(os.path.join(save_path, 'Accuracy_TestDigits.npy'), np.array([acc]))

    #print('accuracy on raw digits', acc)

    #print("\nEvaluate test digits similarity to digits by Logistic regression...")
    pred_probs_test_logreg = logreg_digits.predict_proba(bin_X_test)
    prob_winner_pred_test = pred_probs_test_logreg.max(axis=1)
    mean_qual = np.mean(prob_winner_pred_test)
    #print("Mean quality of test digits", mean_qual, "Std: ", np.std(prob_winner_pred_test))

    ind_winner_pred_test = np.argmax(pred_probs_test_logreg, axis=1)
    test_class, test_counts = np.unique(ind_winner_pred_test, return_counts=True)
    dic = dict(zip(test_class, test_counts))
    #print("sample counts per class",dic)

    mean_qual_d = np.zeros(len(test_class))

    for i in range(0,len(test_class)):
        ind=np.where(ind_winner_pred_test==test_class[i])
        mean_qual_d[i] = np.mean(prob_winner_pred_test[ind])

    qual_d_test = [test_class, mean_qual_d, test_counts]
    np.save(os.path.join(save_path, 'ProbsWinDig_TestDigits.npy'), qual_d_test)

def train_classifier_on_raw_digits(save_path=None):
    if save_path is None or not save_path.endswith('.pkl'): 
        if not os.path.exists(os.path.join('..', 'models', 'MNIST')):
            os.makedirs(os.path.join('..', 'models', 'MNIST'))
        save_path = os.path.join('..', 'models', 'MNIST', 'logreg_MNIST.pkl')

    # get training data
    train, test = preprocess_MNIST()

    bin_X_train = train[0]
    y_train = train[1]

    logreg_digits = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=800, n_jobs=2, random_state=4444, C=0.1)  
    logreg_digits.fit(bin_X_train, y_train)
    
    # save this classifier
    #filename = os.path.join(save_path,'logreg_MNIST.pkl')
    _ = joblib.dump(logreg_digits, save_path)

    return logreg_digits

def get_initial_args(model_path=None, random_seed=None):
    if model_path is None: 
        model_path= os.path.join('..', 'models', 'MNIST', 'initial')

    args={}
    args['n_val'] = 0
    args['epochs'] = (20, 20, 20)
    args['n_vis'] = 400
    args['n_hidden'] = (400, 676)
    args['w_init'] = (0.1, 0.1, 0.1)
    args['vb_init'] = (-1, -1, -1) 
    args['hb_init'] = (-2, -2, -2) 
    args['n_gibbs_steps'] = (1,1,1) 
    args['lr'] =  (np.logspace(-2,-4,20), np.logspace(-1,-4,20), np.logspace(-1,-4,20), np.logspace(-1,-4,20)) 
    args['max_epoch'] = 20
    args['batch_size'] = (1, 1, 1)
    args['l2'] = (0., 0., 0., 0.)
    args['momentum'] = [0.5] * 5 + [0.9] 
    args['sample_v_states'] = True 
    args['dropout'] = None
    args['prune']=False
    if random_seed is not None:
        args['random_seed'] = (random_seed, random_seed, random_seed, random_seed)
    else:
        args['random_seed'] = (1337, 1111, 2222, 3333)
    args['dtype'] = 'float32'
    args['v_shape'] = (20,20)
    args['freeze_weights'] = None
    args['filter_shape'] = [(5,5)]  # for the receptive fields
    args['rbm1_dirpath'] = os.path.join(model_path,'MNIST_DBM_Layer1/')  # / at the end necessary!
    args['rbm2_dirpath'] = os.path.join(model_path,'MNIST_DBM_Layer2/')  # / at the end necessary!
    args['double_rf'] = False

    # RBM 2 related
    args['increase_n_gibbs_steps_every'] = 20

    # DBM related
    args['n_particles'] = 1000  # number of persistent Markov chains
    args['max_mf_updates'] = 50 # maximum number of mean-field updates per weight update
    args['mf_tol'] = 1e-7 # mean-field tolerance
    args['max_norm'] = 6. #maximum norm constraint
    args['sparsity_target'] = 0.2 # desired probability of hidden activation
    args['sparsity_cost'] = (0,0,0) # controls the amount of sparsity penalty 
    args['sparsity_damping']=0.8 # decay rate for hidden activations probs
    args['n_layers']=2 # I added this as a parameter for the model, otherwise it couldn't load a DBM from the disk!
    args['dbm_dirpath']=os.path.join(model_path,'MNIST_InitialDBM/')  # / at the end necessary!

    return args

def compute_accuracy_on_hidden_layer_representations(dbm):
    # get test data
    train, test = preprocess_MNIST()
    bin_X_train = train[0]
    y_train = train[1]

    bin_X_test = test[0]
    y_test = test[1]

    # run on cpu
    config = tf.ConfigProto(
        device_count = {'GPU': 0})
    dbm_pruned._tf_session_config = config

    # compute accuracy on transformed input digits (test set)
    final_train = dbm.transform(bin_X_train) 
    final_test = dbm.transform(bin_X_test) 

    logreg = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=800, n_jobs=2, verbose=10, random_state=4444)  
    logreg.fit(final_train, y_train)

    acc = logreg.score(final_test, y_test)
    print('accuracy of classifier on hidden layer representations.')
    return acc


def evaluate_initial_DBM(dbm, model_path=None, n_samples=60000, sample_every=200, plot=False):
    if model_path is None: 
        model_path=os.path.join('..', 'models', 'MNIST', 'initial')

    params = dbm.get_params()

    nv = params['n_visible_']
    nh1 = params['n_hiddens_'][0]
    nh2 = params['n_hiddens_'][1]

    # check that we have access to a GPU 
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        print("Workspace initialized")
    else:
        print("Consider installing GPU version of TF and running sampling from DBMs on GPU.")

    #run on gpu
    config = tf.ConfigProto(
        device_count = {'GPU': 1})
    dbm._tf_session_config = config

    # sample 
    samples = dbm.sample_gibbs(n_gibbs_steps=sample_every, save_model=False, n_runs=n_samples)

    s_v = samples[:,:nv]
    s_h1 = samples[:,nv:nv+nh1]
    s_h2 = samples[:,nv+nh1:]

    if plot: 
        # plot some exemplary visible layer samples
        rand_ind=np.random.choice(n_samples, 100)
        plt.figure(figsize=(10,10))
        plt.style.use('ggplot')
        im_plot(s_v[rand_ind], shape=(20, 20), imshow_params={'cmap': plt.cm.binary})
    
    # evaluate sample quality and diversity
    logreg = get_classifier_trained_on_raw_digits()
    evaluate_classifier_trained_on_raw_digits_on_generated_samples(s_v, logreg, model_path)

    # evaluate hidden unit representations
    acc = compute_accuracy_on_hidden_layer_representations(dbm)
    np.save(os.path.join(model_path, 'Accuracy_hidden_layer_reps.npy'), np.array([acc]))

    # compute FI of weights
    masks = dbm.get_tf_params(scope='masks')
    rf_mask1 = masks['rf_mask']
    prune_mask1 = masks['prune_mask']
    rf_mask2 = masks['rf_mask_1']
    prune_mask2 = masks['prune_mask_1']

    weights = dbm.get_tf_params(scope='weights')
    W1 = weights['W']
    hb1 = weights['hb']
    W2 = weights['W_1']
    hb2 = weights['hb_1']
    vb = weights['vb']

    temp_mask1 = rf_mask1 * prune_mask1
    samples = np.hstack((s_v, s_h1))
    var_est1, _ = FI_weights_var_heur_estimates(samples, nv, nh1, W1) # call without mask 
    fi_weights1 = var_est1.reshape((nh1,nv)).T * temp_mask1
    fi_weights1[rf_mask1==0]=np.nan # to distinguish non-existing weights from 0 weights

    temp_mask2 = rf_mask2 * prune_mask2
    samples = np.hstack((s_h1, s_h2))
    var_est2, _ = FI_weights_var_heur_estimates(samples, nh1, nh2, W2) # call without mask 
    fi_weights2 = var_est2.reshape((nh2,nh1)).T * temp_mask2

    np.save(os.path.join(model_path, 'RBM1_initial_FI_weights.npy'), fi_weights1)
    np.save(os.path.join(model_path, 'RBM2_initial_FI_weights.npy'), fi_weights2)

def get_initial_DBM(model_path=None, args=None, random_seed=None):
    if model_path is None: 
        model_path=os.path.join('..', 'models', 'MNIST', 'initial')

    if args is None: 
        args = get_initial_args(model_path, random_seed)
    
    if not os.path.exists(args['dbm_dirpath']):
        dbm = train_initial_DBM_on_MNIST(args, random_seed)

    else: 
        rbm1 = load_rbm1(Struct(**args))
        rbm2 = load_rbm2(Struct(**args))
        dbm = load_dbm((rbm1,rbm2), Struct(**args))

    return dbm

def train_initial_DBM_on_MNIST(args=None, random_seed=None):
    # get training data
    train, test = preprocess_MNIST()

    bin_X_train = train[0]
    y_train = train[1]
    bin_X_test = test[0]
    y_test = test[1]

    if args is None: 
        args = get_initial_args(None, random_seed)

    # check that we have access to a GPU 
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        print("Workspace initialized")
    else:
        print("Consider installing GPU version of TF and running sampling from DBMs on GPU.")

    # train first RBM
    rbm1 = make_rbm1(bin_X_train, Struct(**args))

    #transform input with RBM (we need the hidden layer representations to train next RBM)
    Q_train = rbm1.transform(bin_X_train) 
    Q_test = rbm1.transform(bin_X_test) 
    Q_train_bin = make_probs_binary(Q_train)
    Q_test_bin = make_probs_binary(Q_test)

    # train second layer RBM
    rbm2 = make_rbm2(Q_train_bin, Struct(**args))

    G_train = rbm2.transform(Q_train_bin)
    G_test = rbm2.transform(Q_test_bin)
    G_train_bin = make_probs_binary(G_train)
    G_test_bin = make_probs_binary(G_test)  

    # build DBM from the two RBMs and train
    dbm = make_dbm(bin_X_train, None, (rbm1, rbm2), Q_train_bin, G_train_bin, Struct(**args)) 

    return dbm

def create_baseline_classifier(top_folder=None):
    if top_folder is None: 
        top_folder = os.path.join('..', 'models', 'MNIST')
    if not os.path.exists(top_folder):
        os.makedirs(top_folder)

    # baselines of classifier trained on raw digits   
    logreg = get_classifier_trained_on_raw_digits()
    evaluate_classifier_trained_on_raw_digits_on_random_patterns(top_folder, logreg)
    evaluate_classifier_trained_on_raw_digits_on_testdigits(top_folder, logreg)
    return logreg

def create_baseline_DBM(dbm_folder=None, random_seed=None):
    if dbm_folder is None: 
        dbm_folder = os.path.join('..', 'models', 'MNIST', 'initial')

    if not os.path.exists(dbm_folder):
        os.makedirs(dbm_folder)

    # train/get baseline DBM
    dbm = get_initial_DBM(dbm_folder, None, random_seed)
    evaluate_initial_DBM(dbm, dbm_folder)
    return dbm
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'DBM Pruning')
    parser.add_argument('seed', default=42, nargs='?', help='Random seed', type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # logreg = create_baseline_classifier(os.path.join('..', 'models', 'MNIST'))
    dbm = create_baseline_DBM(os.path.join('..', 'models', 'MNIST', 'initial_'+str(args.seed)))





    
