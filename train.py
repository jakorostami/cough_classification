import os
os.chdir("data/custom_cough/")

import argparse

import pandas as pd


import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader


import warnings
warnings.filterwarnings("ignore")

import logging
import datetime
from datetime import datetime, timedelta

from typing import List
from utils.custom_cnn import *
from utils.custom_dataloader import *
from utils.helper_functions import *

import json
with open("config.json") as json_data_file:
    config = json.load(json_data_file)

LABELS = [config["labels"]["TRAIN_LABEL"], config["labels"]["TEST_LABEL"]]
PATHS = [config["paths"]["TRAIN_PATH"],  config["paths"]["TEST_PATH"]]

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

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='---------------- Train a 2D CNN with Mel Spectrogram input data. Made for Imagimob case study by Jako Rostami.')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='Number of epochs to train. Defaults to 1.')
    parser.add_argument('-batchsize', '--batchsize', type=int, default=16, help='Number of batches to pass. Defaults to 16.')
    args = parser.parse_args()
    
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
        
    # Instantiate the training and testing datafetching
    gg = Gatherer(batch_size=BATCH_SIZE)
    train_loader, test_loader = gg.initiatie_data()
    
    # Instantiate the Conv model into GPU and initialize weights
    cnn = SingleCNN2D().to("cuda")
    initialize_weights(cnn)
    
    # Set the binary classification loss - When using logits the sigmoid is built-in and no need to state the sigmoid layer in the model
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(cnn.parameters(), lr=0.001) # Classical optimizer

    # Run the train and test pipeline
    results = train_pipeline(model=cnn,
               train_dataloader=train_loader,
               test_dataloader=test_loader,
               optimizer=optim,
               loss_fn=loss_fn,
               epochs=EPOCHS,
               device="cuda")
    

    print("Training and testing finished. Proceeding to save model.")
    
    date_tracker = datetime.today().strftime("%Y-%m-%d-%H_%M")
    model_dir = "C:/Users/jako/data/custom_cough/models"
    model_name = "noisy_2DCNN_MEL_SP" + "_epochs=" + str(EPOCHS) + \
                "_test-acc=" + str(round(results["test_acc"][-1], 2)) + \
                "_batch=" + str(BATCH_SIZE) + \
                "_" + date_tracker + ".pth"
    
    results_name = "noisy_2DCNN_MEL_SP" + "_epochs=" + str(EPOCHS) + \
                "_test-acc=" + str(round(results["test_acc"][-1], 2)) + \
                "_batch=" + str(BATCH_SIZE) + \
                "_" + date_tracker + ".csv"
                
    pd.DataFrame(results).to_csv(os.path.join("C:/Users/jako/data/custom_cough/", results_name), index=True) # Index will act as the number of epochs

    torch.save(cnn.state_dict(), os.path.join(model_dir, model_name))
    
    print("Model saved as: {}".format(model_name))
    print("Session finished.")
    