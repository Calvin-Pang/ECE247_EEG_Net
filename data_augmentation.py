import numpy as np
import argparse
import yaml
from scipy.signal import savgol_filter
import random
import copy
from dataset.augmentation import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print('Config loaded!')

    dataset_path, label_path = config['dataset'], config['label']
    X, y = np.load(dataset_path), np.load(label_path)
    print('Dataset loaded!')
    print()

    N = len(X)
    if config.get('smooth') is not None:
        X = smooth(X, [config['smooth']['window_size'], config['smooth']['poly_degree']])
        print('Smoothing completed!')
        print()

    X_y =  data_to_dict(X, y)

    X_y_argumented = copy.deepcopy(X_y)
    
    if config.get('jitter') is not None:
        ratio = config['jitter']['ratio']
        loc = config['jitter']['loc']
        sigma = config['jitter']['sigma']
        X_y_sub = copy.deepcopy(X_y)
        if ratio != 1: X_y_sub = np.array(random.choices(X_y, k = int(N * ratio)))
        X_y_argumented += jitter(X_y_sub, [loc, sigma])
        print('Jittering completed!')
        print()
        
    if config.get('scale') is not None:
        ratio = config['scale']['ratio']
        loc = config['scale']['loc']
        sigma = config['scale']['sigma']
        X_y_sub = copy.deepcopy(X_y)
        if ratio != 1: X_y_sub = np.array(random.choices(X_y, k = int(N * ratio)))
        X_y_argumented += scale(X_y_sub, [loc, sigma])
        print('Scaling completed!')
        print()
        
    if config.get('downsample') is not None:
        ratios = config['downsample']['ratios']
        factors = config['downsample']['factors']
        X_y_here = copy.deepcopy(X_y)
        X_y_argumented += downsample(X_y_here, ratios, factors)
        print('Downsampling completed!')
        print()
        
    if config.get('cutmean') is not None:
        ratio = config['cutmean']['ratio']
        factor = config['cutmean']['factor']
        X_y_sub = copy.deepcopy(X_y)
        if ratio != 1: X_y_sub = np.array(random.choices(X_y, k = int(N * ratio)))
        X_y_argumented += cutmean(X_y_sub, factor)
        print('Cutmean completed!')
        print()
    
    X_out, y_out = dict_to_data(X_y_argumented)
    print('Argumented data size:', X_out.shape)
    print('Argumented label size:', y_out.shape)
    print()
    np.save(config['X_save_path'], X_out)
    np.save(config['y_save_path'], y_out)
    print('Argumented data and labels saved!')
    
    

    

    


    