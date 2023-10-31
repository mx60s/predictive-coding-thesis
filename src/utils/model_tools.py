import torch
import torchmetrics
import numpy as np
from torch.utils.data import Dataset

# thank you Marlan

class SequentialFrameDataset(Dataset):
    """
    A dataset that provides sequences of frames in correct temporal order, plus the next step as the prediction target.
    Assumes input is a path to a numpy file containing a list of numpy arrays of size (3, 128, 128)
    """
    def __init__(self, data, transform=None, seq_len=7):
        if isinstance(data, str):
            self.frames = np.load(data)
        elif isinstance(data, np.ndarray):
            self.frames = data
        else:
            raise Exception("Provide path to numpy file or numpy array.")

        #self.frames = np.transpose(self.frames, (0, 3, 1, 2))
        
        self.transform = transform
        self.seq_len = seq_len
        
    def __len__(self):
        # As many possible overlapping sequences of specified length, plus their "predictions"
        return (len(self.frames) // (self.seq_len + 1)) - self.seq_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            # do I even want this functionality with the concat?

        # I'm pretty sure that the dataloader will use len() to grab indexes for this but who knows
        # make more robust in the future
        long_img = np.concatenate(self.frames[idx:idx + self.seq_len], axis=1)
        
        preds = self.frames[idx + self.seq_len + 1]
        
        if self.transform:
            sequences = self.transform(long_img)
            preds = self.transform(preds)
        
        sample = [sequences, preds]
        
        return sample
        

def train(dataloader, model, loss_fn, optimizer, device) -> float:
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


def test(dataloader, model, loss_fn, device) -> float:
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

