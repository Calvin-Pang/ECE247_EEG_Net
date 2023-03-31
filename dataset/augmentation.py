import numpy as np
import argparse
import yaml
from scipy.signal import savgol_filter
import random
import copy
from tqdm import tqdm

def data_to_dict(X, y):
    dict_list = []
    for i in range(len(X)):
        dict_list.append({'data': X[i], 'label': y[i]})
    return dict_list

def dict_to_data(dict_list):
    N = len(dict_list)
    X_out = np.zeros((N, 22, 1000))
    y_out = np.zeros(N)
    for i in range(N):
        X_out[i] = dict_list[i]['data']
        y_out[i] = dict_list[i]['label']
    return X_out, y_out

def smooth(X, params = [11, 3]):
    # params = [window_size, poly_degree]
    # window_size: larger -> smoother (must be odd int)
    # poly_degree: smaller -> smoother (int)
    [window_size, poly_degree] = params
    X_out = np.zeros_like(X)
    (N, H, _) = X.shape
    for i in tqdm(range(N), desc = 'Smoothing...'):
        for j in range(H):
            X_out[i][j] = savgol_filter(X[i][j], window_size, poly_degree)
    return X_out

def jitter(X_y, params = [0, 0.05]):
    # params = [loc, sigma]
    [location, sigma] = params
    X, y = dict_to_data(X_y)
    N = len(X)
    for i in tqdm(range(N), desc = 'Jittering...'):
        for j in range(22):
            noise = np.random.normal(loc = location, scale = sigma, size = 1000)
            X[i][j] += noise
    return data_to_dict(X, y)

def scale(X_y, params = [0, 0.1]):
    # params = [loc, sigma]
    [location, sigma] = params
    X, y = dict_to_data(X_y)
    X_scaled = np.zeros_like(X)
    N = len(X)
    for i in tqdm(range(N), desc = 'Scaling...'):
        for j in range(22):
            noise = np.random.normal(loc = location, scale = sigma, size = 1000)
            X_scaled[i][j] = np.multiply(X[i][j], noise)
    return data_to_dict(X_scaled, y)

def downsample_and_resize(arr, factor, label):
    # Downsample the array
    downsampled_arrs = []
    for i in range(factor):
        downsampled_arrs.append(arr[:, i::factor])
        
    # Resize the downsampled arrays to original size
    resized_arrs = []
    for downsampled_arr in downsampled_arrs:
        resized_arr = np.repeat(downsampled_arr, factor, axis=1)[:, :arr.shape[1]]
        resized_arrs.append({'data': resized_arr, 'label': label})
        
    return resized_arrs

def downsample_one_pair(X_y, ratio, factor):
    N = len(X_y)
    X_y_sub = copy.deepcopy(X_y)
    if ratio != 1: X_y_sub = np.array(random.choices(X_y_sub, k = int(N * ratio)))
    X_y_downsampled = []
    for pair in X_y_sub:
        arr, label = pair['data'], pair['label']
        X_y_downsampled += downsample_and_resize(arr, factor, label)
    return X_y_downsampled

def downsample(X_y, ratios, factors):
    n = len(ratios)
    X_y_downsampled = []
    for i in tqdm(range(n), desc = 'Downsampling...'):
        X_y_downsampled_pair = downsample_one_pair(X_y, ratios[i], factors[i])
        X_y_downsampled += X_y_downsampled_pair
    return X_y_downsampled

def cutmean(X_y, factor = 0.1):
    X, y = dict_to_data(X_y)
    N = len(X)
    num_cut = int(N * factor)
    for i in tqdm(range(N), desc = 'Cutmean...'):
        for j in range(22):
            X_mean = np.mean(X[i][j])
            index = random.choices(range(1000), k = num_cut)
            X[i][j][index] = X_mean
    return data_to_dict(X, y)