import numpy as np
import os
from scipy.stats import entropy
import random
import pickle
import string
random.seed(42)
np.random.seed(42)


def main(path_dict: dict, output_dir: str, n_bootstraps: int = 10000, min_samples: int = 100, max_samples: int = 80000):
    """
    Run bootstrapping for the generated digits: they were classified by a classifier. Estimate the diversity of the generated digits by
    computing the entropy over the distribution of generated digits. Repeatedly take samples of different size from the generated digits.
    
    path_dict: dictionary that contains the path to the results numpy array per model
    output_dir: directory in wich results shall be saved
    n_bootstraps: number of repetitions of sampling per sample size per experiment
    min_samples: minimum sample size
    max_samples: maximum sample size
    """
    my_labels = path_dict.keys()
    if not os.path.exsists(output_dir):
        os.makedirs(output_dir)

    # load the digit counts out of the digit diversity results 
    digit_counts = {}
    for exp in path_dict:
        temp_link = path_dict[exp]
        assert os.path.exists(temp_link), "model path does not exist - abort"
        digit_counts[path_dict[exp]] = np.load(temp_link)[2].flatten().astype(int)

    # now we take samples of different size, with replacement and compute entropy
    frames = np.linspace(min_samples, max_samples, num=100).astype(int)

    # prepare dictionaries
    result_diff_sample_size={}
    # turn it into dictionary
    for exp in path_dict:
        result_diff_sample_size[exp] = {}
        for frame in frames:
            result_diff_sample_size[exp][str(frame)] = {}
            result_diff_sample_size[exp][str(frame)]['mean entropy'] = 0
            result_diff_sample_size[exp][str(frame)]['std entropy'] = 0
            result_diff_sample_size[exp][str(frame)]['frame'] = frame

    for frame in frames:  # how many samples
        for exp in my_labels:  # for each experiment
            counts = digit_counts[exp]

            # generate an array of scalars representing the digit classes according to the counts
            to_sample_from = []
            for dig in range(10):
                to_sample_from.append(np.repeat(dig, counts[dig]))

            to_sample_from = np.concatenate(to_sample_from).ravel()
            np.random.shuffle(to_sample_from)
            entropies = np.zeros(n_bootstraps)

            for iter in range(n_bootstraps):
                sample = np.random.choice(to_sample_from, size=frame, replace=True)
                sample_class, sample_counts = np.unique(sample, return_counts=True)
                entropies[iter] = entropy(sample_counts)  # uses base e

            result_diff_sample_size[exp][str(frame)]['mean entropy'] = np.mean(entropies)
            result_diff_sample_size[exp][str(frame)]['std entropy'] = np.std(entropies)

    f_name = os.path.join(output_dir, 'Bootstraps{}_Resamples{}_to{}.p'.format(n_bootstraps, min_samples, max_samples))
    with open(f_name, 'wb') as fp:
        pickle.dump(result_diff_sample_size, fp)


if __name__ == '__main__':

    print('Start bootstrapping')
    # paths to digit diversity results:
    result_paths = {'Initial': os.path.join('..', 'models', 'MNIST', 'initial', 'ProbsWinDig_Initial_Samples.npy'),
                    'Anti-FI': os.path.join('..', 'models', 'MNIST', 'minimal_models', 'antiFI', 'minimal_model_qualdigits.npy'),
                    'Random': os.path.join('..', 'models', 'MNIST', 'minimal_models', 'random', 'minimal_model_qualdigits.npy'),
                    'Variance FI': os.path.join('..', 'models', 'MNIST', 'minimal_models', 'FI', 'minimal_model_qualdigits.npy'),
                    'Heuristic FI': os.path.join('..', 'models', 'MNIST', 'minimal_models', 'heuristicFI', 'minimal_model_qualdigits.npy'),
                    "|w|": os.path.join('..', 'models', 'MNIST', 'minimal_models', 'w_mag', 'minimal_model_qualdigits.npy')}
    output_dir = os.path.join('..', 'models', 'MNIST', 'minimal_models')
    n_bootstraps = 10000
    min_samples = 100
    max_samples = 80000

    main(result_paths, output_dir, n_bootstraps, min_samples, max_samples)
