import torch
import torchmetrics
import numpy as np
from torch.utils.data import Dataset
import os
from utils.data_processing import map_files_to_chunks

# thank you Marlan

class SequentialFrameDataset(Dataset):
    """
    A dataset that provides sequences of frames in correct temporal order, plus the next step as the prediction target.
    Assumes input is a path to a numpy file containing a list of numpy arrays of size (3, 128, 128)
    """
    def __init__(self, source_directory, target_directory_name, transform=None, seq_len=7):
        target_directory = os.path.join(source_directory, target_directory_name)
        
        if not os.path.exists(target_directory):
            self.length = map_files_to_chunks(source_directory, target_directory, 'frames_', seq_len)
        else:
            print("Assuming already indexed files in", target_directory)
            self.length = len([name for name in os.listdir(target_directory)])
        
        self.data_directory = target_directory
        self.transform = transform
        self.seq_len = seq_len
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        # will this work with multiple idx? does it need to?
        filename = self.data_directory + '/' + str(idx) + '.npy'
        sample = np.load(filename)
        sequence = sample[:-1]
        pred = sample[-1]

        del sample

        seq_list = []

        if self.transform:
            for i in range(self.seq_len):
                seq_list.append(self.transform(sequence[i]))
            pred = self.transform(pred)

        # TODO fix no transform case
        seq_tensor = torch.stack(seq_list)
        
        sample = [seq_tensor, pred]
        
        return sample

class CoordinateDataset(Dataset):
    """
    Essentially the same as the sequential frame dataset, but this time the prediction will be the 
    ground-truth coordinates of the agent for the predicted next frame.
    """
    def __init__(self, source_directory, target_directory_name, transform=None, seq_len=7):
        target_dir_frames = os.path.join(source_directory, target_directory_name, 'frames')
        target_dir_coords = os.path.join(source_directory, target_directory_name, 'coords')
        if not os.path.exists(target_dir_frames):
            self.length = map_files_to_chunks(source_directory, target_dir_frames, 'frames_', seq_len)
        else:
            print("Assuming already indexed frame files in", target_dir_frames)
            self.length = len([name for name in os.listdir(target_dir_frames)])
            
        if not os.path.exists(target_dir_coords):
            map_files_to_chunks(source_directory, target_dir_coords, 'coords_', seq_len)
        else:
            print("Assuming already indexed coords files in", target_dir_coords)
        
        self.frames_directory = target_dir_frames
        self.coords_directory = target_dir_coords
        self.transform = transform
        self.seq_len = seq_len 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_file = self.frames_directory + '/' + str(idx) + '.npy'
        coords_file = self.coords_directory + '/' + str(idx) + '.npy'
        frames = np.load(frame_file)
        coords = np.load(coords_file)
        sequence = frames[:-1]
        pred = coords[-1]

        del frames
        del coords

        seq_list = []
        if self.transform:
            for i in range(self.seq_len):
                seq_list.append(self.transform(sequence[i]))
        
        seq_tensor = torch.stack(seq_list)
        pred_tensor = torch.tensor(pred)
        
        sample = [seq_tensor, pred_tensor]
        
        return sample


def train(dataloader, model, loss_fn, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        #print(X.shape)
        gen = model(X)

        loss = loss_fn(gen, y) 
        loss.backward()
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def test(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            gen = model(X)
            test_loss += loss_fn(gen, y).item()

    test_loss /= num_batches

    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss

def train_no_pred(dataloader, model, loss_fn, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)
        optimizer.zero_grad()
        
        #print(X.shape)
        gen = model(X)

        loss = loss_fn(gen, X) 
        loss.backward()
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def test_no_pred(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            gen = model(X)
            test_loss += loss_fn(gen, X).item()

    test_loss /= num_batches

    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss