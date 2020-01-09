import os
import random
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter

from collections import OrderedDict

import matplotlib.pyplot as plt

# the list of existing scores
#list_scores = ['NSS', 'AUC_Judd', 'AUC_Borji', 'sAUC', 'CC', 'SIM', 'KL', 'IG']
list_scores = ['NSS', 'AUC_Judd', 'AUC_Borji', 'sAUC', 'CC', 'SIM', 'KL']
# a canoncial size to project fixation maps to
#canonical_size = [768, 1024]
canonical_size = [480, 640]
# the sigma for smoothing the fixation maps for a center bias
sigma = 7


def all_fix_map(images):
    '''
        get the folder to the fixation maps and combine them to obtain an all fixation map
    :param images:
    :return:
    '''
    all_fixs = None
    for ind, image in enumerate(images):
        im = scipy.misc.imread(image)
        im = scipy.misc.imresize(im, canonical_size)
        if ind == 0:
            all_fixs = np.zeros(im.shape, dtype=np.float)
        all_fixs += im
    all_fixs[all_fixs > 0.0] = 1.0
    return all_fixs


def get_center_map(all_fix_map):
    '''
        snooth the all fixtion maps using a Gaussian kernel to obtain a center bias map
    :param all_fix_map:
    :return:
    '''
    return gaussian_filter(all_fix_map, sigma=sigma)


def normalize_range(map):
    '''
        normalize a density map so each element be in the range of 0 and 1 such that the maximum value in matrix be 1
    :param map:
    :return:
    '''
    return (map - map.min()) / (map.max() - map.min() + 1e-5)


def compute_nss(saliency, fix_map):
    '''
        Compute NSS score
    :param saliency: saliency map
    :param fix_map: fixation map, which is a binary map of 0, 1 indicating locations of the fixations
    :return:
    '''

    fix_map = fix_map.astype(np.bool)

    if saliency.shape != fix_map.shape:
        saliency = scipy.misc.imresize(saliency, fix_map.shape)

    saliency = (saliency - np.mean(saliency)) / np.std(saliency)
    return np.mean(saliency[fix_map])


def compute_auc_judd(saliency, fix_map, jitter=True):
    '''
        compute the auc judd score
    :param saliency:
    :param fix_map:
    :param jitter:
    :return:
    '''

    if saliency.shape != fix_map.shape:
        saliency = scipy.misc.imresize(saliency, fix_map.shape)

    fix_map = fix_map.flatten().astype(np.bool)
    saliency = saliency.flatten().astype(np.float)

    if jitter:
        jitter = np.random.rand(saliency.shape[0]) / 1e7
        saliency += jitter

    saliecny = normalize_range(saliency)
    Sth = np.sort(saliecny[fix_map])[::-1]

    tp = np.concatenate([[0], np.linspace(0.0, 1.0, Sth.shape[0]), [1]])
    fp = np.zeros((Sth.shape[0]))
    sorted_sal = np.sort(saliecny)

    for ind, th in enumerate(Sth):
        above_threshold = sorted_sal.shape[0] - sorted_sal.searchsorted(th, side='left')
        fp[ind] = (above_threshold-ind) * 1. / (saliecny.shape[0] - Sth.shape[0])
    fp = np.concatenate([[0], fp, [1]])
    return np.trapz(tp, fp)


def compute_auc_borji(saliency, fix_map, n_split=100, step_size=.1):
    '''
        compute the auc borji score
    :param saliency:
    :param fix_map:
    :param n_split:
    :param step_size:
    :return:
    '''

    if saliency.shape != fix_map.shape:
        saliency = scipy.misc.imresize(saliency, fix_map.shape)
    fix_map = fix_map.flatten().astype(np.bool)

    saliency = normalize_range(saliency.flatten().astype(np.float))
    sal_on_fix = np.sort(saliency[fix_map])

    r = np.random.randint(0, saliency.shape[0], (sal_on_fix.shape[0], n_split))
    rand_sal = saliency[r]
    auc = np.zeros(n_split)

    for i in range(n_split):
        cur_sal = rand_sal[:,i]
        sorted_cur_fix = np.sort(cur_sal)

        max_val = np.maximum(cur_sal.max(), sal_on_fix.max())
        tmp_treshold = np.arange(0, max_val, step_size)[::-1]
        tp = np.zeros(tmp_treshold.shape[0])
        fp = np.zeros(tmp_treshold.shape[0])

        for ind, ths in enumerate(tmp_treshold):
            tp[ind] = (sal_on_fix.shape[0] - sal_on_fix.searchsorted(ths, side='left'))*1./sal_on_fix.shape[0]
            fp[ind] = (sorted_cur_fix.shape[0] - sorted_cur_fix.searchsorted(ths, side='left'))*1./sal_on_fix.shape[0]
        tp = np.concatenate([[0], tp, [1]])
        fp = np.concatenate([[0], fp, [1]])
        auc[i] = np.trapz(tp, fp)
    return np.mean(auc)


def compute_cc(saliency, gt_sal):
    '''
        compute corelation coefficient
    :param saliency:
    :param gt_sal:
    :return:
    '''
    if saliency.shape != gt_sal.shape:
        saliency = scipy.misc.imresize(saliency, gt_sal.shape)
    saliency = (saliency - saliency.mean())/(saliency.std())
    gt_sal = (gt_sal - gt_sal.mean())/(gt_sal.std())
    return np.corrcoef(saliency.flat, gt_sal.flat)[0, 1]


def compute_sim(saliency, gt_sal):
    '''
        compute the similarity score
    :param saliency:
    :param gt_sal:
    :return:
    '''
    if saliency.shape != gt_sal.shape:
        saliency = scipy.misc.imresize(saliency, gt_sal.shape)
    saliency = saliency.astype(np.float)
    gt_sal = gt_sal.astype(np.float)

    saliency = normalize_range(saliency)
    saliency = saliency / saliency.sum()

    gt_sal = normalize_range(gt_sal)
    gt_sal = gt_sal / gt_sal.sum()
    diff = np.minimum(saliency, gt_sal)
    return np.sum(diff)


def compute_kl(saliency, gt_sal):
    '''
        compute KL-divergence score between saliency and saliency ground truth
    :param saliency:
    :param gt_sal:
    :return:
    '''
    if saliency.shape != gt_sal.shape:
        saliency = scipy.misc.imresize(saliency, gt_sal.shape)
    eps = np.finfo(np.float).eps
    # normalzie the saliency and groundtruth to be distributions sum to 1
    saliency = saliency.astype(np.float)
    gt_sal = gt_sal.astype(np.float)
    saliency = saliency / saliency.sum()
    gt_sal = gt_sal / gt_sal.sum()

    return np.sum(gt_sal * np.log(eps + (gt_sal / (saliency + eps))))


def compute_ig(saliency, fix_map, base_sal):
    '''
        compute the information gain IG
    :param saliency:
    :param fix_map:
    :param base_sal:
    :return:
    '''

    if saliency.shape != fix_map.shape:
        saliency = scipy.misc.imresize(saliency, fix_map.shape)
    if base_sal.shape != fix_map.shape:
        base_sal = scipy.misc.imresize(base_sal, fix_map.shape)
    eps = np.finfo(float).eps

    locs = fix_map.astype(np.bool).flatten()

    saliency = saliency.astype(np.float32).flatten()
    base_sal = base_sal.astype(np.float32).flatten()
    saliency = saliency / saliency.sum()
    base_sal = base_sal / base_sal.sum()

    return np.mean(np.log2(eps+saliency[locs])-np.log2(eps+base_sal[locs]))


def compute_auc_shuffled(saliency, fix_map, base_map, n_split=100, step_size=.1):
    '''
        computing shuffled AUC
    :param saliency:
    :param fix_map:
    :param base_map:
    :param n_split:
    :param step_size:
    :return:
    '''
    if saliency.shape != fix_map.shape:
        saliency = scipy.misc.imresize(saliency, fix_map.shape)

    saliency = saliency.flatten().astype(np.float)
    base_map = base_map.flatten().astype(np.float)
    fix_map = fix_map.flatten().astype(np.bool)

    saliency = normalize_range(saliency)
    sal_on_fix = np.sort(saliency[fix_map])

    ind = np.where(base_map > 0.0)[0]
    n_fix = sal_on_fix.shape[0]
    n_fix_oth = np.minimum(n_fix, ind.shape[0])
    rand_fix = np.zeros((n_fix_oth, n_split))

    for i in range(n_split):
        rand_ind = random.sample(list(ind), n_fix_oth)
        rand_fix[:,i] = saliency[rand_ind]

    auc = np.zeros(n_split)

    for i in range(n_split):
        cur_fix = np.sort(rand_fix[:, i])
        max_val = np.maximum(cur_fix.max(), sal_on_fix.max())
        tmp_treshold = np.arange(0, max_val, step_size)[::-1]
        tp = np.zeros(tmp_treshold.shape[0])
        fp = np.zeros(tmp_treshold.shape[0])
        for ind, ths in enumerate(tmp_treshold):
            tp[ind] = (sal_on_fix.shape[0] - sal_on_fix.searchsorted(ths, side='left')) * 1. / n_fix
            fp[ind] = (cur_fix.shape[0] - cur_fix.searchsorted(ths, side='left')) * 1. / n_fix_oth
        tp = np.concatenate([[0], tp, [1]])
        fp = np.concatenate([[0], fp, [1]])
        auc[i] = np.trapz(tp, fp)
    return np.mean(auc)

def get_image_list(folder):
    '''
        load images from folders
    :param folder:
    :return:
    '''
    images = os.listdir(folder)
    assert len(images) > 0

    return images

def get_images(folder):
    '''
        load images from folders
    :param folder:
    :return:
    '''
    images = os.listdir(folder)
    assert len(images) > 0

    return [os.path.join(folder, file) for file in images]


def compute_scores(saliency_folder, density_map_folder, fix_map_folder):
    '''
        compute the scores for all the scores
    :param saliency_folder:
    :param density_map_folder:
    :param fix_map_folder:
    :return:
    '''

    file_names = sorted(get_image_list(saliency_folder))

    saliency_files = [os.path.join(saliency_folder, file) for file in file_names]
    density_files = [os.path.join(density_map_folder, '{}.png'.format(file[:-4])) for file in file_names]
    fix_map_files = [os.path.join(fix_map_folder, '{}.png'.format(file[:-4])) for file in file_names]
    total_fix_map = all_fix_map(fix_map_files)

    res = OrderedDict()
    for s in list_scores:
        res[s] = list()

    ls = len(saliency_files)
    lm = len(density_files)
    l = min(ls, lm)
    for ind in range(l):
        saliency = scipy.misc.imread(saliency_files[ind])
        #saliency = np.mean(saliency, axis=2)
        #saliency = scipy.misc.imread('center2.png')
        density = scipy.misc.imread(density_files[ind])
        fix_map = scipy.misc.imread(fix_map_files[ind])

        if sum(sum(fix_map)) == 0:
            print('GT is zero')
            continue

        if density.shape != canonical_size:
            density = scipy.misc.imresize(density, canonical_size)
        density = normalize_range(density)

        if fix_map.shape != canonical_size:
            fix_map = scipy.misc.imresize(fix_map, canonical_size)

        fix_map[fix_map > 0.0] = 1.0

        if saliency.shape != canonical_size:
            saliency = scipy.misc.imresize(saliency, canonical_size)
        saliency = normalize_range(saliency)
        if sum(sum(saliency)) == 0:
            print('SAL is zero')
            continue


        res['NSS'].append(compute_nss(saliency, fix_map))

        res['AUC_Judd'].append(compute_auc_judd(saliency, fix_map))
        res['AUC_Borji'].append(compute_auc_borji(saliency, fix_map))
        base_map = total_fix_map - fix_map
        base_map[base_map < 0.0] = 0.0
        res['sAUC'].append(compute_auc_shuffled(saliency, fix_map, base_map))
        res['CC'].append(compute_cc(saliency, density))
        res['SIM'].append(compute_sim(saliency, density))
        res['KL'].append(compute_kl(saliency, density))

        #base_map = total_fix_map - fix_map
        #base_map[base_map < 0.0] = 0.0
        #base_map = get_center_map(base_map)
        #base_map = np.load('train_center.npy')
        #base_map = base_map - density
        #base_map[base_map < 0.0] = 0.0
        #res['IG'].append(compute_ig(saliency, density, base_map))

    return res, saliency_files


def compute_scores_w_center(saliency_folder, density_map_folder, fix_map_folder):
    '''
        compute the scores for all the scores
    :param saliency_folder:
    :param density_map_folder:
    :param fix_map_folder:
    :return:
    '''

    #saliency_files = sorted(get_images(saliency_folder))
    saliency_files = sorted(get_images(saliency_folder),  key=lambda s: int(os.path.basename(s)[:-4]))
    #density_files = sorted(get_images(density_map_folder),  key=lambda s: int(s[:-4]))
    density_files = sorted(get_images(density_map_folder),  key=lambda s: int(os.path.basename(s)[:-4]))
    fix_map_files = sorted(get_images(fix_map_folder),  key=lambda s: int(os.path.basename(s)[:-4]))
    #fix_map_files = sorted(get_images(fix_map_folder))
    total_fix_map = all_fix_map(fix_map_files)

    res = OrderedDict()
    for s in list_scores:
        res[s] = list()

    ls = len(saliency_files)
    lm = len(density_files)
    l = min(ls, lm)
    for ind in range(l):
        saliency = scipy.misc.imread(saliency_files[ind])


        center = scipy.misc.imread('center_train.png')
        center = np.mean(center, axis=2)
        density = scipy.misc.imread(density_files[ind])
        fix_map = scipy.misc.imread(fix_map_files[ind])

        if sum(sum(fix_map)) == 0:
            print('GT is zero')
            continue

        if density.shape != canonical_size:
            density = scipy.misc.imresize(density, canonical_size)
        density = normalize_range(density)

        if fix_map.shape != canonical_size:
            fix_map = scipy.misc.imresize(fix_map, canonical_size)

        fix_map[fix_map > 0.0] = 1.0

        if saliency.shape != canonical_size:
            saliency = scipy.misc.imresize(saliency, canonical_size)
        saliency = normalize_range(saliency)
        center = 1 + normalize_range(center)
        saliency = np.multiply(saliency, center)
        saliency = normalize_range(saliency)
        if sum(sum(saliency)) == 0:
            print('SAL is zero')
            continue


        res['NSS'].append(compute_nss(saliency, fix_map))

        res['AUC_Judd'].append(compute_auc_judd(saliency, fix_map))
        res['AUC_Borji'].append(compute_auc_borji(saliency, fix_map))
        base_map = total_fix_map - fix_map
        base_map[base_map < 0.0] = 0.0
        res['sAUC'].append(compute_auc_shuffled(saliency, fix_map, base_map))
        res['CC'].append(compute_cc(saliency, density))
        res['SIM'].append(compute_sim(saliency, density))
        res['KL'].append(compute_kl(saliency, density))

        #base_map = total_fix_map - fix_map
        #base_map[base_map < 0.0] = 0.0
        #base_map = get_center_map(base_map)
        #base_map = np.load('train_center.npy')
        #base_map = base_map - density
        #base_map[base_map < 0.0] = 0.0
        #res['IG'].append(compute_ig(saliency, density, base_map))

    return res, saliency_files
