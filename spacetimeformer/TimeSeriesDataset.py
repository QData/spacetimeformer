import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data_folder, context_length, forecast_length):
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.data_files = self.load_csv_files(data_folder)
        self.cumulative_lengths = self.get_cumulative_lengths(self.data_files)

    def load_csv_files(self, folder):
        all_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')]
        return [pd.read_csv(file, index_col=0).values for file in all_files]  # Treat the first column as index


    def get_cumulative_lengths(self, data_files):
        lengths = [len(file) - (self.context_length + self.forecast_length) for file in data_files]
        return np.cumsum([0] + lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        file_index = np.where(self.cumulative_lengths > idx)[0][0] - 1
        within_file_idx = idx - self.cumulative_lengths[file_index]
        context = self.data_files[file_index][within_file_idx:within_file_idx+self.context_length]
        forecast = self.data_files[file_index][within_file_idx+self.context_length:within_file_idx+self.context_length+self.forecast_length]
        return torch.tensor(context, dtype=torch.float), torch.tensor(forecast, dtype=torch.float)
# Create DataLoaders for each dataset
train_dataset = TimeSeriesDataset(data_folder='./data/train', context_length=10, forecast_length=10)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TimeSeriesDataset(data_folder='./data/test', context_length=10, forecast_length=10)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

oos_dataset = TimeSeriesDataset(data_folder='./data/oos', context_length=10, forecast_length=10)
oos_dataloader = DataLoader(oos_dataset, batch_size=32, shuffle=False)

# Example of iterating over a DataLoader
for context, forecast in train_dataloader:
    # Model training code here
    1+1
