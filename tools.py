import numpy as np
import utils, pdb

def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

def transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n
        P[num_classes-1, num_classes-1] = 1. - n
    return P


def fit(X, num_classes, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10, noise_type='symmetric',feature_size=28*28):
    clean_train_labels = train_labels[:, np.newaxis]

    if noise_type == 'symmetric':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_symmetric(clean_train_labels,
                                                noise=noise_rate, random_state= random_seed, nb_classes=num_classes)
        #print(noisy_labels.shape)
        #rint(type(noisy_labels))

    elif noise_type == 'flip':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_pairflip(clean_train_labels,
                                                    noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
    elif noise_type == 'asymmetric':
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_asymmetric(clean_train_labels,
                                                    noise=noise_rate, random_state=random_seed, nb_classes=num_classes)

    elif noise_type == 'instance':
        noisy_labels, real_noise_rate, clean_data,clean_lables = utils.noisify_instance(train_images, train_labels,noise_rate=noise_rate,feature_size=feature_size)

        noisy_labels = np.array(noisy_labels)
        noisy_labels = noisy_labels.squeeze()

        return train_images, noisy_labels, train_labels
        #return  clean_data,clean_lables,None


    noisy_labels = noisy_labels.squeeze()

    return train_images,  noisy_labels, train_labels    #transition_matrix
    #return train_set, val_set, train_labels, val_labels, transition_matrix
    