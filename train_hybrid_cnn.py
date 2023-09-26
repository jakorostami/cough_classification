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
from utils.custom_hybrid_2d_3d_cnn import *
from utils.helper_functions import *

from sklearn.model_selection import KFold


import json
with open("config.json") as json_data_file:
    config = json.load(json_data_file)

LABELS = [config["labels"]["TRAIN_LABEL"], config["labels"]["TEST_LABEL"]]
PATHS = [config["paths"]["TRAIN_TOPO_3D"],  config["paths"]["TRAIN_PATH"], config["paths"]["TEST_TOPO_3D"], config["paths"]["TEST_PATH"]]

class MultiInputData(Dataset):
    
    def __init__(self, topo_path, melspec_path, annotations_file):
        self.topo_path = topo_path
        self.melspec_path = melspec_path
        self.annotations = pd.read_csv(annotations_file)
        
    
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        topo_3d, mel_spec = self._get_topo_path(idx), self._get_spectro_path(idx)
        label = self._get_audio_sample_label(idx)
        
        topo = torch.tensor(torch.load(topo_3d)).unsqueeze_(0)
        mel = torch.tensor(torch.load(mel_spec)).unsqueeze_(0)
        
        return topo.float(), mel.float(), label
        
    
    def _get_spectro_path(self, idx):
        path = os.path.join(self.melspec_path, self.annotations.iloc[idx, 2])
        return path
    
    def _get_topo_path(self, idx):
        
        path = os.path.join(self.topo_path, self.annotations.iloc[idx, 2])
        return path
    
    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 1]
    


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
    
def training_grounds(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     loss_fn: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     device: str = "cuda") -> torch.Tensor:
    
    model.train()
    
    train_loss, train_acc = 0, 0
    all_preds, all_targets = [], []
    
    for batch, (input3d, input2d, target) in enumerate(dataloader):
        input3d, input2d, target = input3d.to(device), input2d.to(device), target.to(device)
        target = target.float()
        
        prediction = model(input3d, input2d)
        prediction = prediction.squeeze(dim=1)
        loss = loss_fn(prediction, target)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        pred_class = (torch.sigmoid(prediction) > 0.5).float()
        all_preds.extend(pred_class.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        train_acc += (pred_class == target).sum().item() / len(prediction)
    
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, precision, recall, f1


def testing_grounds(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     loss_fn: torch.nn.Module,
                     device: str = "cuda", 
                     mc_dropout: bool = False,
                     mc_runs: int = 5) -> torch.Tensor:
    
    if mc_dropout:
        model.train() # Need to put it in training mode for inference because of gradients
    else:
        model.eval()
    
    test_loss, test_acc = 0, 0
    all_preds, all_targets = [], []
    
    with torch.inference_mode():
        for batch, (input3d, input2d, target) in enumerate(dataloader):
            input3d, input2d, target = input3d.to(device), input2d.to(device), target.to(device)
            target = target.float()
            
            
            if mc_dropout:
                mc_outputs = [model(input3d, input2d).squeeze(dim=1) for _ in range(mc_runs)]
                test_preds = torch.mean(torch.stack(mc_outputs), dim=0)
            else:                
                test_preds = model(input3d, input2d)
                test_preds = test_preds.squeeze(dim=1)
            
            loss = loss_fn(test_preds, target)
            test_loss += loss.item()
            
            test_pred_class = (torch.sigmoid(test_preds) > 0.5).float()
            all_preds.extend(test_pred_class.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            test_acc += (test_pred_class == target).sum().item() / len(test_preds)

    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
       
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, precision, recall, f1



def train_pipeline(model: torch.nn.Module,
                   train_dataloader: torch.utils.data.DataLoader,
                   test_dataloader: torch.utils.data.DataLoader,
                #    val_dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: torch.nn.Module = nn.BCELoss(),
                   epochs: int = 1,
                   device: str = "cuda"):
    
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               "train_prec": [],
               "train_rec": [],
               "train_f1": [],
               "test_prec": [],
               "test_rec": [],
               "test_f1": []}
    
    
    for epoch in range(epochs):
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = training_grounds(model=model,
                                                 dataloader=train_dataloader,
                                                 loss_fn=loss_fn,
                                                 optimizer=optimizer,
                                                 device=device
                                                 )
        scheduler.step()
        
        test_loss, test_acc, test_prec, test_rec, test_f1 = testing_grounds(model=model,
                                              dataloader=test_dataloader,
                                              loss_fn=loss_fn,
                                              device=device
                                              )
        
        # val_loss, val_acc = validation_grounds(model=model,
        #                                        dataloader=val_dataloader,
        #                                        loss_fn=loss_fn,
        #                                        device=device)
        
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | Test precision {test_prec:.4f} | Test recall {test_rec:.4f} | Test F1: {test_f1:.4f}")

        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_prec"].append(train_prec)
        results["train_rec"].append(train_rec)
        results["train_f1"].append(train_f1)
        results["test_prec"].append(test_prec)
        results["test_rec"].append(test_rec)
        results["test_f1"].append(test_f1)
        
    
    return results



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
    # gg = Gatherer(batch_size=BATCH_SIZE)
    train_loader, test_loader = MultiInputData(PATHS[0], PATHS[1], LABELS[0]), MultiInputData(PATHS[2], PATHS[3], LABELS[1])
    
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
        hybrid_cnn = HybridCNN().to("cuda")
        hybrid_cnn.apply(reset_weights)
        initialize_weights(hybrid_cnn)

        loss_fn = nn.BCEWithLogitsLoss()
        optim = torch.optim.Adam(hybrid_cnn.parameters(), lr=0.01, weight_decay=1e-5)
        # optim = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
        
        
        for epoch in range(0, EPOCHS):
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            train_loss, train_acc, train_prec, train_rec, train_f1 = training_grounds(hybrid_cnn, trainloader, loss_fn, optim)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Precision: {train_prec:.4f}, Train Recall: {train_rec:.4f}, Train F1: {train_f1:.4f}")
            results[epoch] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1
            }
        
        
        test_loss, test_acc, test_prec, test_rec, test_f1 = testing_grounds(hybrid_cnn, testloader, loss_fn, mc_dropout=mc_dropout)
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
