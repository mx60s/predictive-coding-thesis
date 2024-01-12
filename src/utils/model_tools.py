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
        self.source_directory = source_directory
        self.transform = transform
        self.seq_len = seq_len
        self.file_pairs = []
        self.sample_indices = []

        frames_files = sorted([f for f in os.listdir(source_directory) if f.startswith('frames_')])
        coords_files = sorted([f for f in os.listdir(source_directory) if f.startswith('coords_')])

        for f_file in frames_files:
            f_file.index('_')
            suffix = f_file[f_file.index('_'):]
            c_file = 'coords' + suffix
            if c_file in coords_files:
                self.file_pairs.append((f_file, c_file))

            frames = np.load(os.path.join(self.source_directory, f_file))
            num_samples = len(frames) - (self.seq_len + 1) + 1

            for i in range(num_samples):
                self.sample_indices.append((len(self.file_pairs) - 1, i))
        
        
    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.sample_indices[idx]
        frame_file, coord_file = self.file_pairs[file_idx]

        frames = np.load(os.path.join(self.source_directory, frame_file))[sample_idx:sample_idx + self.seq_len + 1]
        coords = np.load(os.path.join(self.source_directory, coord_file))[sample_idx:sample_idx + self.seq_len + 1]
        
        sequence = frames[:-1]
        pred = frames[-1]
        
        headings = coords[:, 2]

        # change in heading between steps, in degrees
        differences = []
        for i in range(self.seq_len):
            diff = headings[i+1] - headings[i]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            differences.append(diff)

        del frames
        del coords

        seq_list = []

        if self.transform:
            for i in range(self.seq_len):
                seq_list.append(self.transform(sequence[i]))
            pred = self.transform(pred)

        heading_tensor = torch.FloatTensor(differences)

        # TODO fix no transform case
        seq_tensor = torch.stack(seq_list)
        
        sample = [(seq_tensor, heading_tensor), pred]
        
        return sample

def train(dataloader, model, loss_fn, optimizer, device) -> float:
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


def test(dataloader, model, loss_fn, device) -> float:
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