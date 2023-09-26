import os
os.chdir("data/custom_cough/")

import argparse


import pandas as pd


import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


import warnings
warnings.filterwarnings("ignore")

import logging
import datetime
from datetime import datetime, timedelta

from typing import List
from utils.custom_topological_cnn import *
from utils.custom_dataloader import *
from utils.helper_functions import *

from sklearn.model_selection import KFold


import json
with open("config.json") as json_data_file:
    config = json.load(json_data_file)

LABELS = [config["labels"]["TRAIN_LABEL"], config["labels"]["TEST_LABEL"]]
PATHS = [config["paths"]["TRAIN_TOPO"],  config["paths"]["TEST_TOPO"]]

# trainload = CoughData(TRAIN_PATH, TRAIN_LBL)
# testload = CoughData(TEST_PATH, TEST_LBL)
# trainer = DataLoader(trainload, batch_size=BATCH_SIZE)
# tester = DataLoader(testload, batch_size=BATCH_SIZE)


class Gatherer:
    def __init__(self, 
                 batch_size: int, 
                 torch_dataloader: torch.utils.data.DataLoader = DataLoader,
                 dataset: torch.utils.data.Dataset = CoughData,
                 datapaths: List[str] = PATHS,
                 datalabels: List[str] = LABELS,
                 log_dir: str = "./logs/"):
        
        self.dtformat = datetime.now().strftime("%Y-%m-%d")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        report_name = log_dir + "logging_report_" + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".log"
        handler = logging.FileHandler(report_name)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.batch_size=BATCH_SIZE
        self.dataset = dataset
        self.torch_dataloader = torch_dataloader
        self.datapaths = datapaths
        self.datalabels = datalabels
    
    def initiatie_data(self):
        self.logger.info("Using batch size: {}".format(self.batch_size))
        self.logger.info("Instantiating training dataset")
        self.trainload = self.dataset(self.datapaths[0],
                                      self.datalabels[0])
        self.logger.info("Finished train. Instantiating testing dataset.")
        self.testload = self.dataset(self.datapaths[1],
                                      self.datalabels[1])
        
        self.logger.info("Training and testing loaders finished.\
                        Now instantiating training and testing dataloaders")
        
        self.trainer, self.tester = self.torch_dataloader(self.trainload, batch_size=self.batch_size, shuffle=True), self.torch_dataloader(self.testload, batch_size=self.batch_size, shuffle=False)
        
        self.logger.info("Finished dataloaders.")
        
        return self.trainer, self.tester
    

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='---------------- Train a 2D CNN with topological signals with Takens embeddings input. Made for Imagimob case study by Jako Rostami.')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='Number of epochs to train. Defaults to 1.')
    parser.add_argument('-batchsize', '--batchsize', type=int, default=16, help='Number of batches to pass. Defaults to 16.')
    parser.add_argument("-montecarlo", "--montecarlo", type=bool, default=False, help="Use Monte Carlo dropout or not. Defaults to False.")
    parser.add_argument("-mc_runs", "--mc_runs", type=int, default=5, help="Number of Monte Carlo runs. Defaults to 5.")
    parser.add_argument("-mc_dropout", "--mc_dropout", type=float, )
    args = parser.parse_args()
    
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
        
    # Instantiate the training and testing datafetching
    gg = Gatherer(batch_size=BATCH_SIZE)
    train_loader, test_loader = CoughData(PATHS[0], LABELS[0]), CoughData(PATHS[1], LABELS[1])
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    # For fold results
    results = {}
    
    dataset = ConcatDataset([train_loader, test_loader])
    
    kfold = KFold(n_splits=5, shuffle=True)
    mc_dropout = True
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"Fold: {fold}")
        print("---"*10)
        
        fold_str = "fold_" + str(fold)
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        testloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_subsampler)
        
    
    
        # Instantiate the Conv model into GPU and initialize weights
        cnn = SingleCNN2DTopo(mc_dropout=mc_dropout).to("cuda")
        cnn.apply(reset_weights)
        initialize_weights(cnn)
        optim = torch.optim.Adam(cnn.parameters(), lr=0.001) # Classical optimizer
        
        
        for epoch in range(0, EPOCHS):
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            train_loss, train_acc, train_prec, train_rec, train_f1 = training_grounds(cnn, trainloader, loss_fn, optim)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Precision: {train_prec:.4f}, Train Recall: {train_rec:.4f}, Train F1: {train_f1:.4f}")
            results[epoch] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1
            }
        
        
        test_loss, test_acc, test_prec, test_rec, test_f1 = testing_grounds(cnn, testloader, loss_fn, mc_dropout=mc_dropout)
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | Test precision {test_prec:.4f} | Test recall {test_rec:.4f} | Test F1: {test_f1:.4f}")
        
        results[fold] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1
        }

    
    
 
    print("---"*10)


print("Training and testing finished. Proceeding to save model.")

date_tracker = datetime.today().strftime("%Y-%m-%d-%H_%M")
model_dir = "C:/Users/jako/data/custom_cough/models"
# model_name = "noisy_2DCNN_TOPO" + "_epochs=" + str(EPOCHS) + \
#             "_test-acc=" + str(round(results["test_acc"][-1], 2)) + \
#             "_batch=" + str(BATCH_SIZE) + \
#             "_" + date_tracker + ".pth"

# results_name = "noisy_2DCNN_TOPO" + "_epochs=" + str(EPOCHS) + \
#             "_test-acc=" + str(round(results["test_acc"][-1], 2)) + \
#             "_batch=" + str(BATCH_SIZE) + \
#             "_" + date_tracker + ".csv"


            
# pd.DataFrame(results).to_csv(os.path.join("C:/Users/jako/data/custom_cough/models", results_name), index=True) # Index will act as the number of epochs
pd.DataFrame(results).T.to_csv(os.path.join("C:/Users/jako/data/custom_cough/models", "testing_kfold.csv"))

# torch.save(cnn.state_dict(), os.path.join(model_dir, model_name))

print("Model saved as: {}".format("BAJS.pth"))
print("Session finished.")
