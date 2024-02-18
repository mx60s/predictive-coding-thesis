import torch
import torchmetrics
import numpy as np
import pickle
from torch.utils.data import Dataset
import os
from utils.data_processing import map_files_to_chunks


class AutoencoderDataset(Dataset):
    def __init__(self, source_directory, transform=None):
        self.sorted_frames = []

        frames_list = []

        frames_files = sorted([f for f in os.listdir(source_directory) if f.startswith('frames_')])
        print(frames_files)
        for f_file in frames_files:
            frames = np.load(os.path.join(source_directory, f_file))
            frames_list.append(frames)

        self.sorted_frames = torch.stack([transform(frame) for stack in frames_list for frame in stack])
    
    def __len__(self):
        return len(self.sorted_frames)

    def __getitem__(self, idx):
        return self.sorted_frames[idx]


class SequentialFrameDataset(AutoencoderDataset):
    """
    A dataset that provides sequences of frames in correct temporal order, plus the next step as the prediction target.
    Assumes input is a path to a numpy file containing a list of numpy arrays of size (3, 128, 128)
    """
    def __init__(self, source_directory, transform=None, seq_len=7):
        super().__init__(source_directory, transform)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sorted_frames) - self.seq_len

    def __getitem__(self, idx):
        sequence = self.sorted_frames[idx:idx + self.seq_len + 1]
        pred = self.sorted_frames[idx + self.seq_len + 1]

        return [sequence, pred]
        

class HeadingDataset(SequentialFrameDataset):
    """
    A dataset that provides sequences of frames in correct temporal order, plus the next step as the prediction target. It also provides the change in direction at each step.
    Assumes input is a path to a numpy file containing a list of numpy arrays of size (3, 128, 128)
    """
    def __init__(self, source_directory, transform=None, seq_len=7):
        super().__init__(source_directory, transform, seq_len)
        
        self.sorted_headings = []

        # Collect the coordinates info associated with each frame
        coords_list = []
        coords_files = sorted([f for f in os.listdir(source_directory) if f.startswith('coords_')])
        
        for c_file in coords_files:
            coords = np.load(os.path.join(source_directory, c_file))
            coords_list.append(coords)

        coords_list = [coord for stack in coords_list for coord in stack]

        # Calculate heading displacement per step in degrees
        headings = coords_list[:, 2]

        for i in range(1, len(headings)):
            diff = headings[i] - headings[i-1]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            self.sorted_headings.append(diff)

        # Convert to tensor 
        self.sorted_headings = [torch.from_numpy(heading) for heading in self.sorted_headings]

    def __getitem__(self, idx):
        frame_sequence = self.sorted_frames[idx:idx + self.seq_len + 1]
        pred = self.sorted_frames[idx + self.seq_len + 1]

        heading_sequence = self.sorted_headings[idx:idx + self.seq_len + 1]

        return [(frame_sequence, heading_sequence), pred]


class CoordinateDataset(AutoencoderDataset):
    def __init__(self, source_directory, transform=None):
        super().__init__(source_directory, transform)
        
        self.sorted_coords = []

        # Collect the coordinates info associated with each frame
        coords_list = []
        coords_files = sorted([f for f in os.listdir(source_directory) if f.startswith('coords_')])
        print(coords_files)
        for c_file in coords_files:
            coords = np.load(os.path.join(source_directory, c_file)).astype(np.float32)
            coords_list.append(coords)

        self.sorted_coords = [torch.from_numpy(coord) for stack in coords_list for coord in stack]

    def __getitem__(self, idx):
        return [self.sorted_frames[idx], self.sorted_coords[idx]]

# Functions for training

def train_heading(dataloader, model, loss_fn, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        x_img, x_head = X
        x_img, x_head, y = x_img.to(device), x_head.to(device), y.to(device)
        optimizer.zero_grad()
        
        #print(X.shape)
        #gen, weights = model(X)
        gen = model(x_img, x_head)
        
        loss = loss_fn(gen, y) 
        loss.backward()
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def test_heading(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            x_img, x_head = X
            x_img, x_head, y = x_img.to(device), x_head.to(device), y.to(device)
            #gen, weights = model(X)
            gen = model(x_img, x_head)
            test_loss += loss_fn(gen, y).item()

    test_loss /= num_batches

    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss

def train(dataloader, model, loss_fn, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        #print(X.shape)
        #gen, weights = model(X)
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
            X, y = X.to(device), y.to(device)
            #gen, weights = model(X)
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
    for batch, X in enumerate(dataloader):
        X = X.to(device)
        optimizer.zero_grad()
        
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
        for X in dataloader:
            X = X.to(device)
            gen = model(X)
            test_loss += loss_fn(gen, X).item()

    test_loss /= num_batches

    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss