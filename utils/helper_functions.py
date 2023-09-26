import torch.nn as nn
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from manim import *


def training_grounds(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     loss_fn: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     device: str = "cuda") -> torch.Tensor:
    
    model.train()
    
    train_loss, train_acc = 0, 0
    all_preds, all_targets = [], []
    
    for batch, (input, target) in enumerate(dataloader):
        input, target = input.to(device), target.to(device)
        target = target.float()
        
        prediction = model(input)
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
        for batch, (input, target) in enumerate(dataloader):
            input, target = input.to(device), target.to(device)
            target = target.float()
            
            
            if mc_dropout:
                mc_outputs = [model(input).squeeze(dim=1) for _ in range(mc_runs)]
                test_preds = torch.mean(torch.stack(mc_outputs), dim=0)
            else:                
                test_preds = model(input)
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



def validation_grounds(model: torch.nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        loss_fn: torch.nn.Module,
                        device: str = "cuda") -> torch.Tensor:
    
    model.eval()  # Set the model to evaluation mode
    
    val_loss, val_acc = 0, 0
    
    # Disable gradient computation for efficiency since we won't be updating the model
    with torch.inference_mode():  
        for batch, (input, target) in enumerate(dataloader):
            input, target = input.to(device), target.to(device)
            target = target.float()
            
            val_preds = model(input)
            val_preds = val_preds.squeeze(dim=1)
            
            loss = loss_fn(val_preds, target)
            val_loss += loss.item()
            
            # Convert model output to class labels
            val_pred_class = (torch.sigmoid(val_preds) > 0.5).float()
            val_acc += (val_pred_class == target).sum().item() / len(val_preds)
        
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    
    return val_loss, val_acc



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


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
        
        
class TensorSlicing(ThreeDScene):
    def construct(self):
        # Create a 3D tensor (for illustration, we'll use a cube)
        tensor = Cube()
        self.set_camera_orientation(phi=510 * DEGREES, theta=330 * DEGREES, gamma=270 * DEGREES)

        # Add the tensor to the scene
        self.add(tensor)
        
        # Add boundary gridlines to the tensor
        boundaries = [-1, 1]
        for b in boundaries:
            # Lines parallel to X-axis
            self.add(Line(start=(-1, b, -1), end=(1, b, -1), color=GREY))
            self.add(Line(start=(-1, b, 1), end=(1, b, 1), color=GREY))
            # Lines parallel to Y-axis
            self.add(Line(start=(b, -1, -1), end=(b, 1, -1), color=GREY))
            self.add(Line(start=(b, -1, 1), end=(b, 1, 1), color=GREY))
            # Lines parallel to Z-axis
            self.add(Line(start=(-1, -1, b), end=(-1, 1, b), color=GREY))
            self.add(Line(start=(1, -1, b), end=(1, 1, b), color=GREY))

        # Connecting lines for the corners
        for x in boundaries:
            for y in boundaries:
                self.add(Line(start=(x, y, -1), end=(x, y, 1), color=GREY))
        for x in boundaries:
            for z in boundaries:
                self.add(Line(start=(x, -1, z), end=(x, 1, z), color=GREY))
        for y in boundaries:
            for z in boundaries:
                self.add(Line(start=(-1, y, z), end=(1, y, z), color=GREY))



        # Define the number of slices and the distance between them
        num_slices = 6
        distance = 2 / num_slices  # Assuming the cube has a side length of 2
        
        
        # Fill the cube with random points
        num_points = 100
        for _ in range(num_points):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
            point = Dot(point=(x, y, z), radius=0.025, color=BLACK)
            self.add(point)
            
            
        # For each slice, create a square and animate it moving through the tensor
        for i in range(num_slices):
            slice = Square(side_length=2, fill_opacity=0.5, fill_color=BLUE)
            slice.shift(OUT * (-1 + i * distance))
            
            # Animate the slice moving through the tensor
            self.play(FadeIn(slice))
            self.wait(0.2)
            self.remove(slice)
        
        self.wait()
        
def cut_when_needed(signal):
    if len(signal) > (7*16000):
        signal = signal[:7*16000]
    else:
        signal = signal
    return signal

def stretch_when_needed(signal):
    signal_length = len(signal)
    if signal_length < (7*16000):
        num_missing = (7*16000) - signal_length
        last = (0, num_missing)
        signal = torch.nn.functional.pad(torch.tensor(signal), last)
    else:
        signal = signal
    return signal



def create_3d_tensor_from_pca(pca_data, shape=(24, 24, 24)):
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
    
    return tensor_3d


def training_grounds_hybrid(model: torch.nn.Module,
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


def testing_grounds_hybrid(model: torch.nn.Module,
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



def train_pipeline_hybrid(model: torch.nn.Module,
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
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = training_grounds_hybrid(model=model,
                                                 dataloader=train_dataloader,
                                                 loss_fn=loss_fn,
                                                 optimizer=optimizer,
                                                 device=device
                                                 )
        scheduler.step()
        
        test_loss, test_acc, test_prec, test_rec, test_f1 = testing_grounds_hybrid(model=model,
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