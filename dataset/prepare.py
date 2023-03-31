import numpy as np
import os
import argparse
from PIL import Image
from tqdm import tqdm

def train_val(dataset_path, save_dir, split_ratio):
    imgs = np.load(dataset_path)
    N, H, W = imgs.shape
    num_train = round(N * split_ratio[0])
    if not(os.path.exists(os.path.join(save_dir, 'train'))): os.makedirs(os.path.join(save_dir, 'train'))
    if not(os.path.exists(os.path.join(save_dir, 'val'))): os.makedirs(os.path.join(save_dir, 'val'))
    for i, img in tqdm(enumerate(imgs), desc = 'processing traing and val data...'):
        sample_id = str(i + 1)
        set_name = 'train' if i < num_train else 'val'
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        for j in range(50):
            patch_id = str(j + 1)
            name = sample_id + '_' + patch_id + '.npy'
            patch = img[ : , j * 20 : (j * 20 + 20)]
            np.save(os.path.join(save_dir, set_name, name), patch)
            

def test_data(dataset_path, save_dir):
    imgs = np.load(dataset_path)
    if not(os.path.exists(os.path.join(save_dir, 'test'))): os.makedirs(os.path.join(save_dir, 'test'))
    for i, img in tqdm(enumerate(imgs), desc = 'processing test data...'):
        sample_id = str(i + 1)
        set_name = 'test'
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        for j in range(50):
            patch_id = str(j + 1)
            name = sample_id + '_' + patch_id + '.npy'
            patch = img[ : , j * 20 : (j * 20 + 20)]
            np.save(os.path.join(save_dir, set_name, name), patch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type = str)
    parser.add_argument('--split-ratio')
    parser.add_argument('--save-dir', type = str)
    parser.add_argument('--mode', type = str)
    args = parser.parse_args()

    if not(os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    if args.mode == 'train_val':
        train, val = list(map(float, args.split_ratio.split(',')))
        split_ratio_list = [train, val]
        train_val(args.dataset_path, args.save_dir, split_ratio_list)
        
    elif args.mode == 'test':
        test_data(args.dataset_path, args.save_dir)