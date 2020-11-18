# helper functions to initialize, fit and load RBMs and a 2-layer DBM
import os
from bm.rbm.rbm import BernoulliRBM, logit_mean
from bm.dbm import DBM
import numpy as np
import tensorflow as tf

def load_rbm1(args):
    if os.path.isdir(args.rbm1_dirpath):
        print("\nLoading RBM #1 ...\n\n")
        rbm1 = BernoulliRBM.load_model(args.rbm1_dirpath)
        return rbm1
    else:
        print("model could not be found!")

def load_rbm2(args):
    if os.path.isdir(args.rbm2_dirpath):
        print("\nLoading RBM #2 ...\n\n")
        rbm2 = BernoulliRBM.load_model(args.rbm2_dirpath)
        return rbm2
    else:
        print("model could not be found!")

def load_rbm3(args):
    if os.path.isdir(args.rbm3_dirpath):
        print("\nLoading RBM #3 ...\n\n")
        rbm3 = BernoulliRBM.load_model(args.rbm3_dirpath)
        return rbm3
    else:
        print("model could not be found!")

def load_dbm_withoutRBMs(args):
    if os.path.isdir(args.dbm_dirpath):
        print("\nLoading DBM ...\n\n")
        dbm = DBM.load_model(args.dbm_dirpath)
        return dbm
    else:
        print("model could not be found!")

def load_dbm(rbms, args):
    if os.path.isdir(args.dbm_dirpath):
        print("\nLoading DBM ...\n\n")
        dbm = DBM.load_model(args.dbm_dirpath)
        dbm.load_rbms(rbms)  # !!!
        return dbm
    else:
        print("model could not be found!")

def make_rbm1(X, args):
    if os.path.isdir(args.rbm1_dirpath):
        print("\nLoading RBM #1 ...\n\n")
        rbm1 = BernoulliRBM.load_model(args.rbm1_dirpath)
    else:
        print("\nTraining RBM #1 ...\n\n")

        if not hasattr(args, 'double_rf'): 
          args.double_rf = False

        rbm1 = BernoulliRBM(n_visible=args.n_vis,
                            n_hidden=args.n_hidden[0],
                            W_init= args.w_init[0],
                            vb_init=args.vb_init[0],
                            hb_init=args.hb_init[0],
                            n_gibbs_steps=args.n_gibbs_steps[0],
                            learning_rate=args.lr[0],
                            momentum=args.momentum,
                            max_epoch=args.epochs[0],
                            batch_size=args.batch_size[0],
                            l2=args.l2[0],
                            sample_h_states=True,
                            sample_v_states=True,
                            sparsity_cost=0.,
                            dbm_first=True,  # !!!
                            metrics_config=dict(
                                msre=True,
                                pll=True,
                                train_metrics_every_iter=500,
                            ),
                            verbose=True,
                            #display_filters=30,
                            #display_hidden_activations=24,
                            v_shape=args.v_shape,
                            freeze_weights = args.freeze_weights,
                            prune = args.prune,
                            filter_shape = args.filter_shape[0],
                            double_rf = args.double_rf,
                            random_seed=args.random_seed[0],
                            dtype='float32',
                            tf_saver_params=dict(max_to_keep=1),
                            model_path=args.rbm1_dirpath)
        #run on cpu
        config = tf.ConfigProto(
          device_count = {'GPU': 0}
          )
        rbm1._tf_session_config = config
        rbm1.fit(X)
    return rbm1

def init_rbm1(args):
    if os.path.isdir(args.rbm1_dirpath):
        print("\nLoading RBM #1 ...\n\n")
        rbm1 = BernoulliRBM.load_model(args.rbm1_dirpath)
    else:
        print("\nInitializing RBM #1 ...\n\n")

        if not hasattr(args, 'double_rf'): 
          args.double_rf = False

        rbm1 = BernoulliRBM(n_visible=args.n_vis,
                            n_hidden=args.n_hidden[0],
                            W_init= args.w_init[0],
                            vb_init=args.vb_init[0],
                            hb_init=args.hb_init[0],
                            n_gibbs_steps=args.n_gibbs_steps[0],
                            learning_rate=args.lr[0],
                            momentum=args.momentum,
                            max_epoch=args.epochs[0],
                            batch_size=args.batch_size[0],
                            l2=args.l2[0],
                            sample_h_states=True,
                            sample_v_states=True,
                            sparsity_cost=0.,
                            dbm_first=True,  # !!!
                            metrics_config=dict(
                                msre=True,
                                pll=True,
                                train_metrics_every_iter=500,
                            ),
                            double_rf = args.double_rf,
                            verbose=True,
                            #display_filters=30,
                            #display_hidden_activations=24,
                            v_shape=args.v_shape,
                            freeze_weights = args.freeze_weights,
                            prune = args.prune,
                            filter_shape = args.filter_shape[0],
                            random_seed=args.random_seed[0],
                            dtype='float32',
                            tf_saver_params=dict(max_to_keep=1),
                            model_path=args.rbm1_dirpath)
        #run on cpu
        config = tf.ConfigProto(
          device_count = {'GPU': 0}
          )
        rbm1._tf_session_config = config
        rbm1.init()
    return rbm1

def make_rbm2(Q,args):
    if os.path.isdir(args.rbm2_dirpath):
        print("\nLoading RBM #2 ...\n\n")
        rbm2 = BernoulliRBM.load_model(args.rbm2_dirpath)
    else:
        print("\nTraining RBM #2 ...\n\n")
        epochs = args.epochs[1]
        n_every = args.increase_n_gibbs_steps_every

        n_gibbs_steps = np.arange(args.n_gibbs_steps[1],
                        args.n_gibbs_steps[1] + epochs / n_every)
        learning_rate = args.lr[1] / np.arange(1, 1 + epochs / n_every)
        n_gibbs_steps = np.repeat(n_gibbs_steps, n_every)
        learning_rate = np.repeat(learning_rate, n_every)

        rbm2 = BernoulliRBM(n_visible=args.n_hidden[0],
                            n_hidden=args.n_hidden[1],
                            W_init=args.w_init[1],
                            vb_init=args.vb_init[1],
                            hb_init=args.hb_init[1],
                            n_gibbs_steps=n_gibbs_steps,
                            learning_rate=learning_rate,
                            momentum=args.momentum,
                            max_epoch=max(args.epochs[1], n_every),
                            batch_size=args.batch_size[1],
                            l2=args.l2[1],
                            sample_h_states=True,
                            sample_v_states=True,
                            sparsity_cost=0.,
                            dbm_last=True,  # !!!
                            metrics_config=dict(
                                msre=True,
                                pll=True,
                                train_metrics_every_iter=500,
                            ),
                            verbose=True,
                            #display_filters=0,
                            #display_hidden_activations=24,
                            random_seed=args.random_seed[1],
                            v_shape = (20,20), # what's the v_shape in this case?
                            freeze_weights = args.freeze_weights,
                            prune = args.prune,
                            #filter_shape =  (10,10), #,args.filter_shape[1],
                            dtype='float32',
                            tf_saver_params=dict(max_to_keep=1),
                            model_path=args.rbm2_dirpath)
        #run on cpu
        config = tf.ConfigProto(
          device_count = {'GPU': 0}
          )
        rbm2._tf_session_config = config
        rbm2.fit(Q)
    return rbm2

def init_rbm2(args):
    if os.path.isdir(args.rbm2_dirpath):
        print("\nLoading RBM #2 ...\n\n")
        rbm2 = BernoulliRBM.load_model(args.rbm2_dirpath)
    else:
        print("\nInitializing RBM #2 ...\n\n")
        epochs = args.epochs[1]
        n_every = args.increase_n_gibbs_steps_every

        n_gibbs_steps = np.arange(args.n_gibbs_steps[1],
                        args.n_gibbs_steps[1] + epochs / n_every)
        learning_rate = args.lr[1] / np.arange(1, 1 + epochs / n_every)
        n_gibbs_steps = np.repeat(n_gibbs_steps, n_every)
        learning_rate = np.repeat(learning_rate, n_every)

        rbm2 = BernoulliRBM(n_visible=args.n_hidden[0],
                            n_hidden=args.n_hidden[1],
                            W_init=args.w_init[1],
                            vb_init=args.vb_init[1],
                            hb_init=args.hb_init[1],
                            n_gibbs_steps=n_gibbs_steps,
                            learning_rate=learning_rate,
                            momentum=args.momentum,
                            max_epoch=max(args.epochs[1], n_every),
                            batch_size=args.batch_size[1],
                            l2=args.l2[1],
                            sample_h_states=True,
                            sample_v_states=True,
                            sparsity_cost=0.,
                            dbm_last=True,  # !!!
                            metrics_config=dict(
                                msre=True,
                                pll=True,
                                train_metrics_every_iter=500,
                            ),
                            verbose=True,
                            #display_filters=0,
                            #display_hidden_activations=24,
                            random_seed=args.random_seed[1],
                            v_shape = (20,20), # what's the v_shape in this case?
                            freeze_weights = args.freeze_weights,
                            prune = args.prune,
                            #filter_shape =  (10,10), #,args.filter_shape[1],
                            dtype='float32',
                            tf_saver_params=dict(max_to_keep=1),
                            model_path=args.rbm2_dirpath)
        #run on cpu
        config = tf.ConfigProto(
          device_count = {'GPU': 0}
          )
        rbm2._tf_session_config = config
        rbm2.init()
    return rbm2

def make_dbm(X_train, X_val, rbms, Q, G, args):
    if os.path.isdir(args.dbm_dirpath):
        print("\nLoading DBM ...\n\n")
        dbm = DBM.load_model(args.dbm_dirpath)
        dbm.load_rbms(rbms)  # !!!
    else:
        print("\nTraining DBM ...\n\n")

        dbm = DBM(rbms=rbms,
                  n_layers = args.n_layers,
                  n_particles=args.n_particles,
                  v_particle_init=X_train[:args.n_particles].copy(),
                  h_particles_init=(Q[:args.n_particles].copy(),
                                    G[:args.n_particles].copy()),
                                    #D[:args.n_particles].copy()), # for three layers
                  n_gibbs_steps=args.n_gibbs_steps[2],
                  max_mf_updates=args.max_mf_updates,
                  mf_tol=args.mf_tol,
                  learning_rate= args.lr[2],#np.geomspace(args.lr[2], 5e-6, 400),
                  momentum= args.momentum, #np.geomspace(0.5, 0.9, 10),
                  max_epoch=args.epochs[2],
                  batch_size=args.batch_size[2],
                  l2=args.l2[2],
                  max_norm=args.max_norm,
                  sample_v_states=True,
                  sample_h_states=(True, True, True),
                  sparsity_target=args.sparsity_target,
                  sparsity_cost=args.sparsity_cost,
                  sparsity_damping=args.sparsity_damping,
                  train_metrics_every_iter=400,
                  val_metrics_every_epoch=2,
                  random_seed=args.random_seed[2],
                  verbose=True,
                  display_filters=0,
                  display_particles=0,
                  v_shape=(20, 20),
                  dtype='float32',
                  tf_saver_params=dict(max_to_keep=1),
                  model_path=args.dbm_dirpath)
        #run on cpu
        config = tf.ConfigProto(
          device_count = {'GPU': 0}
          )
        dbm._tf_session_config = config
        dbm.fit(X_train, X_val)
    return dbm

def init_dbm(X_train, X_val, rbms, Q, G, args):
    if os.path.isdir(args.dbm_dirpath):
        print("\nLoading DBM ...\n\n")
        dbm = DBM.load_model(args.dbm_dirpath)
        dbm.load_rbms(rbms)  # !!!
    else:
        print("\nInitializing DBM ...\n\n")

        dbm = DBM(rbms=rbms,
                  n_layers = args.n_layers,
                  n_particles=args.n_particles,
                  v_particle_init=X_train[:args.n_particles].copy(),
                  h_particles_init=(Q[:args.n_particles].copy(),
                                    G[:args.n_particles].copy()),
                                    #D[:args.n_particles].copy()),
                  n_gibbs_steps=args.n_gibbs_steps[2],
                  max_mf_updates=args.max_mf_updates,
                  mf_tol=args.mf_tol,
                  learning_rate= args.lr[2],#np.geomspace(args.lr[2], 5e-6, 400),
                  momentum= args.momentum, #np.geomspace(0.5, 0.9, 10),
                  max_epoch=args.epochs[2],
                  batch_size=args.batch_size[2],
                  l2=args.l2[2],
                  max_norm=args.max_norm,
                  sample_v_states=True,
                  sample_h_states=(True, True, True),
                  sparsity_target=args.sparsity_target,
                  sparsity_cost=args.sparsity_cost,
                  sparsity_damping=args.sparsity_damping,
                  train_metrics_every_iter=400,
                  val_metrics_every_epoch=2,
                  random_seed=args.random_seed[2],
                  verbose=True,
                  display_filters=0,
                  display_particles=0,
                  v_shape=(20, 20),
                  dtype='float32',
                  tf_saver_params=dict(max_to_keep=1),
                  model_path=args.dbm_dirpath)
        #run on cpu
        config = tf.ConfigProto(
          device_count = {'GPU': 0}
          )
        dbm._tf_session_config = config
        dbm.init()
    return dbm



def init_dbm_without_particles(X_train, X_val, rbms, args):
    if os.path.isdir(args.dbm_dirpath):
        print("\nLoading DBM ...\n\n")
        dbm = DBM.load_model(args.dbm_dirpath)
        dbm.load_rbms(rbms)  # !!!
    else:
        print("\nInitializing DBM ...\n\n")

        dbm = DBM(rbms=rbms,
                  n_layers = args.n_layers,
                  n_particles=args.n_particles,
                  #v_particle_init=X_train[:args.n_particles].copy(),
                  #h_particles_init=(Q[:args.n_particles].copy(),
                  #                  G[:args.n_particles].copy()),
                  #                  D[:args.n_particles].copy()),
                  n_gibbs_steps=args.n_gibbs_steps[2],
                  max_mf_updates=args.max_mf_updates,
                  mf_tol=args.mf_tol,
                  learning_rate= args.lr[2],#np.geomspace(args.lr[2], 5e-6, 400),
                  momentum= args.momentum, #np.geomspace(0.5, 0.9, 10),
                  max_epoch=args.epochs[2],
                  batch_size=args.batch_size[2],
                  l2=args.l2[2],
                  max_norm=args.max_norm,
                  sample_v_states=True,
                  sample_h_states=(True, True, True),
                  sparsity_target=args.sparsity_target,
                  sparsity_cost=args.sparsity_cost,
                  sparsity_damping=args.sparsity_damping,
                  train_metrics_every_iter=400,
                  val_metrics_every_epoch=2,
                  random_seed=args.random_seed[2],
                  verbose=True,
                  display_filters=0,
                  display_particles=0,
                  v_shape=(20, 20),
                  dtype='float32',
                  tf_saver_params=dict(max_to_keep=1),
                  model_path=args.dbm_dirpath)
        #run on cpu
        config = tf.ConfigProto(
          device_count = {'GPU': 0}
          )
        dbm._tf_session_config = config
        dbm.init()
    return dbm


def init_rbm(args): # initializes standard RBM
#     if os.path.isdir(args.model_dirpath):
#         print ("\nLoading model ...\n\n")
#         rbm = BernoulliRBM.load_model(args.model_dirpath)
#     else:
#     print ("\nTraining model ...\n\n")
    rbm = BernoulliRBM(n_visible=args.n_visible,
                       n_hidden=args.n_hidden,
                       W_init=args.w_init,
                       vb_init=args.vb_init,
                       hb_init=args.hb_init,
                       prune= args.prune,
                       freeze_weights = args.freeze_weights,
                       filter_shape = args.filter_shape,
                       n_gibbs_steps=args.n_gibbs_steps,
                       learning_rate=args.lr,
#                        momentum=np.geomspace(0.5, 0.9, 8),
                       momentum=args.momentum,
                       max_epoch=args.epochs,
                       batch_size=args.batch_size,
                       l2=args.l2,
                       sample_v_states=args.sample_v_states,
                       sample_h_states=True,
                       dropout=args.dropout,
                       sparsity_target=args.sparsity_target,
                       sparsity_cost=args.sparsity_cost,
                       sparsity_damping=args.sparsity_damping,
                       metrics_config=dict(
                           msre=False,
                           pll=False,
                           feg=False,
                           train_metrics_every_iter=10,
                           val_metrics_every_epoch=1,
                           feg_every_epoch=4,
                           n_batches_for_feg=10,
                       ),
                       verbose=True,
                       random_seed=args.random_seed,
                       dtype=args.dtype,
                       tf_saver_params=dict(max_to_keep=1),
                       model_path=args.model_dirpath)
    rbm.init()
    return rbm
