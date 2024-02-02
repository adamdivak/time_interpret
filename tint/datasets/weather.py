import numpy as np
import pandas as pd
import os
import random
import torch as th
import tqdm
import pickle as pkl

from tint.datasets.dataset import DataModule

file_dir = os.path.dirname(__file__)

EPS = 1e-5

class Weather(DataModule):
    def __init__(
        self, 
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "weather",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        n_folds: int = None,
        fold: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            n_folds=n_folds,
            fold=fold,
            num_workers=num_workers,
            seed=seed,
        )
        
        # Init mean and std
        self._mean = None
        self._std = None
        
    def download(self, prop_train: float = 0.8):  
        # Some choices
        datapoints = 1000
        threshold = 0
        random.seed(42)
        
        # Print the data directory and the current files in it
        print("Data directory: ", self.data_dir)
        print("Files in data directory: ", os.listdir(self.data_dir))
        
        # load data
        file_location = os.path.join(self.data_dir, 'weather data NL subset.txt')
        data = pd.read_csv(file_location, skiprows=31, skipinitialspace=True)
        # Now remove rows with missing values
        data = data.dropna()
        data = data.replace(-1,0)
        
        # Get some descriptive statistics
        weather_stations = data['# STN'].unique()
        
        # Now create the arrays for the data and the labels
        features = len(data.columns) - 2
        
        x = np.zeros((datapoints, features, 48))
        y = np.zeros((datapoints,))
        
        # Now populate the arrays
        for i in range(datapoints):
            # Get a random weather station
            station = random.choice(weather_stations)
            
            # Now determine the number of datapoints for this station
            station_data = data[data['# STN'] == station]
            datapoints_length = len(station_data)
            
            # Now get a random set of 49 datapoints, thus random number between 0 and datapoints_length - 49
            start = random.randint(0, datapoints_length - 49)
            end = start + 49
            
            # Now get the data for this station
            station_data = station_data.iloc[start:end,:]
            
            # Now populate the x array
            x[i, :, :] = station_data.iloc[:-1, 2:].transpose()
            
            # Now get the label
            label = station_data["RH"].iloc[-1]
            y[i] = 1 if label > threshold else 0
          
        # Now that we have filled the data array and labels, we export it to a pickle file for easy analysis
        with open(
            os.path.join(
                self.data_dir, "weather_data.pkl"
            ),
            "wb",
        ) as f:
            pkl.dump((x, y), f)
        
        
        samples = [(x[i,:, :], y[i]) for i in range(datapoints)]   
        # Now split the data into train and test
        train_size = int(datapoints * prop_train)
        train_samples = samples[:train_size]
        test_samples = samples[train_size:]
        
        # Save preprocessed data
        with open(
            os.path.join(
                self.data_dir, "train_weather.pkl"
            ),
            "wb",
        ) as f:
            pkl.dump(train_samples, f)
        with open(
            os.path.join(
                self.data_dir, "test_weather.pkl"
            ),
            "wb",
        ) as f:
            pkl.dump(test_samples, f)
        
        
    def prepare_data(self):
        if not os.path.exists(
            os.path.join(self.data_dir, "train_weather.pkl")
        ) or not os.path.join(
            self.data_dir, "test_weather.pkl"
        ):
            self.download()
            
    def preprocess(self, split: str = "train") -> dict:
        file = os.path.join(self.data_dir, f"{split}_")
        with open(file + "weather.pkl", "rb") as fp:
            data = pkl.load(fp)
            
        features = th.Tensor([x for (x, y) in data]).transpose(1, 2)
        
        labels = th.Tensor([y for (x, y) in data])
        
        # Compute mean and std
        if split == "train":
            self._mean = features.mean(dim=(0,1), keepdim=True)
            self._std = features.std(dim=(0,1), keepdim=True)
        else:
            assert split == "test", "split must be train or test"
            
        assert (
            self._mean is not None
        ), "You must call preprocess('train') first"
        
        # Normalise
        features = (features - self._mean) / (self._std + EPS)
        
        return {"x": features.float(),
                "y": labels.long()}
        