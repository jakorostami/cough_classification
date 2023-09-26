import os
os.chdir("data/custom_cough/")


import joblib
from joblib import Parallel, delayed
import librosa
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.decomposition import PCA
from gtda.time_series import SingleTakensEmbedding

def create_3d_tensor_from_pca(pca_data, shape=(22, 4, 4)):
    # Determine the range for each axis
    mins = pca_data.min(axis=0)
    maxs = pca_data.max(axis=0)
    
    # Create the 3D tensor filled with zeros
    tensor_3d = np.zeros(shape)
    
    # Determine the step size for each dimension
    step_sizes = (maxs - mins) / np.array(shape)
    
    # Distribute the PCA points into the tensor
    for point in pca_data:
        # Calculate the voxel's index for each point
        idx = ((point - mins) / step_sizes).astype(int)
        
        # Clip to ensure we don't exceed the shape due to floating point inaccuracies
        idx = np.clip(idx, 0, np.array(shape) - 1)
        
        # Increment the voxel where the point falls
        tensor_3d[tuple(idx)] += 1
    
    return torch.tensor(tensor_3d)


# pca = PCA(n_components=3)

train_labels = pl.read_csv("train_full/full_train_labels.csv")

wav_paths = ["custom_cough/cough", "custom_cough/noise_cough_augmented", 
             
             "custom_cough/no_cough", "custom_cough/noise_no_cough_augmented"]


def cut_when_needed(signal: np.array):
    if len(signal) > (7*16000):
        signal = signal[:7*16000]
    else:
        signal = signal
    return signal

def stretch_when_needed(signal: np.array):
    signal_length = len(signal)
    if signal_length < (7*16000):
        num_missing = (7*16000) - signal_length
        last = (0, num_missing)
        signal = torch.nn.functional.pad(torch.tensor(signal), last)
    else:
        signal = signal
    return signal


# tks = SingleTakensEmbedding(time_delay=1, dimension=100, stride=1, n_jobs=-1, parameters_type="fixed")

# dir_path = "C:/Users/jako/data/custom_cough/topological_3D_test"

def process_row(row, paths_check, dir_path, pca, tks):
    for path in paths_check:
        fullpath = os.path.join(path, row)

        if os.path.exists(fullpath):
            y, sr = librosa.load(fullpath, sr=None)
            y = cut_when_needed(y)
            y = stretch_when_needed(y)
            y = tks.fit_transform(y)
            y = pca.fit_transform(y)
            y = create_3d_tensor_from_pca(y, shape=(24, 24, 24))

            newname = row[:-4] + ".pt"

            torch.save(y, os.path.join(dir_path, newname))

            return f"Processed {row}"
    return f"Skipped {row}"

def process_group(args):
    group, train_labels, wav_paths, dir_path, pca, tks = args
    data = train_labels.filter(train_labels["class_type"] == group)
    paths_check = wav_paths[:2] if group == 1 else wav_paths[2:]

    results = [process_row(row, paths_check, dir_path, pca, tks) for row in data["id"]]
    return results

if __name__ == "__main__":
    train_labels = pd.read_csv("C:/Users/jako/data/custom_cough/test_full/full_test_labels.csv")
    wav_paths = ["C:/Users/jako/data/custom_cough/cough", "C:/Users/jako/data/custom_cough/noise_cough_augmented", 
             
             "C:/Users/jako/data/custom_cough/no_cough", "C:/Users/jako/data/custom_cough/noise_no_cough_augmented"]
    dir_path = "C:/Users/jako/data/custom_cough/topological_3D_test"

    pca = PCA(n_components=3)
    tks = SingleTakensEmbedding(time_delay=1, dimension=100, stride=1, n_jobs=-1, parameters_type="fixed")

    for group in train_labels["class_type"].unique():
        data = train_labels[train_labels["class_type"] == group]["id"].tolist()
        paths_check = wav_paths[:2] if group == 1 else wav_paths[2:]
        
        # Use joblib to parallelize the operation
        Parallel(n_jobs=-1)(delayed(process_row)(row, paths_check, dir_path, pca, tks) for row in data)
