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
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.pairs, self.cumulative_sizes = self._find_pairs()
        self.transform = transform

    def _find_pairs(self):
        files = os.listdir(self.directory)
        frames_files = [f for f in files if f.startswith("frames_")]
        pairs = []
        cumulative_sizes = [0]
        for frame_file in frames_files:
            timestamp = frame_file[len("frames_"):]
            coord_file = f"coords_{timestamp}"
            if coord_file in files:
                pairs.append((frame_file, coord_file))
                frame_count = len(np.load(os.path.join(self.directory, frame_file)))
                cumulative_sizes.append(cumulative_sizes[-1] + frame_count)
        return pairs, cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        file_index = next(i for i, total in enumerate(self.cumulative_sizes) if total > idx)
        frame_file, coord_file = self.pairs[file_index - 1]
        frame_index = idx - self.cumulative_sizes[file_index - 1]

        frame_path = os.path.join(self.directory, frame_file)
        coord_path = os.path.join(self.directory, coord_file)
        
        frames = np.load(frame_path)
        coords = np.load(coord_path)

        frame = frames[frame_index]
        coord = coords[frame_index][:-1] # for now, exclude the heading direction

        if self.transform:
            frame = self.transform(frame)

        coord = torch.from_numpy(coord).type(torch.FloatTensor)
        
        return frame, coord

class HeadingDataset(Dataset):
    """
    A dataset that provides sequences of frames in correct temporal order, plus the next step as the prediction target.
    Assumes input is a path to a numpy file containing a list of numpy arrays of size (3, 128, 128)
    """
    def __init__(self, source_directory, transform=None, seq_len=7):
        frames_directory = os.path.join(source_directory, 'frames')
        coords_directory = os.path.join(source_directory, 'coords')
        
        if not os.path.exists(frames_directory): # for now, assuming that if one exists, the other should
            self.length = map_files_to_chunks(source_directory, frames_directory, 'frames_', seq_len)
            map_files_to_chunks(source_directory, coords_directory, 'coords_', seq_len)
        else:
            print("Assuming already indexed files")
            self.length = len([name for name in os.listdir(frames_directory)])

        self.frames_directory = frames_directory
        self.coords_directory = coords_directory
        self.transform = transform
        self.seq_len = seq_len
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # will this work with multiple idx? does it need to?
        frame_file = f'{self.frames_directory}/' + str(idx) + '.npy'
        coord_file = f'{self.coords_directory}/' + str(idx) + '.npy'
        frames = np.load(frame_file)
        coords = np.load(coord_file)
        sequence = frames[:-1]
        pred = frames[-1]
        headings = coords[:, 2]

        # change in heading between steps, in degrees
        for i in range(self.seq_len):
            headings[i] = headings[i+1] - headings[i]

        del frames
        del coords

        seq_list = []

        if self.transform:
            for i in range(self.seq_len):
                seq_list.append(self.transform(sequence[i]))
            pred = self.transform(pred)

        heading_tensor = torch.from_numpy(headings[:-1])

        # TODO fix no transform case
        seq_tensor = torch.stack(seq_list)
        
        sample = [(seq_tensor, heading_tensor), pred]
        
        return sample

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
            X,y = X.to(device), y.to(device)
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