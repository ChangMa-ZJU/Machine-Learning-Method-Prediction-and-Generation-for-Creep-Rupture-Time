import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample.values)

class DataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=transforms.Compose([ToTensor()]), training_porcentage = 1.0, shuffle = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.data = pd.read_csv(csv_file).head(100000)
        self.file = pd.read_csv(csv_file)
        if (shuffle):
            self.file = shuffle(self.file)
        self.data = self.file.head(int(self.file.shape[0]*training_porcentage)) 
        self.test_data = self.file.tail(int(self.file.shape[0]*(1-training_porcentage)))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if self.transform:
            item = self.transform(item)
        return item

    def get_columns(self):
        return self.data.columns

class DataAtts():
    def __init__(self, file_name):
        if file_name == "data/data_filter_nortar.csv":
            self.message = " data is processed by RAE and the target is normalized"
            self.class_name = "RT"
            self.values_names = {"feature_name",  "target"}
            self.class_len = 29  # 28+1
            self.fname = "model_filter_nontrun"
        
        elif file_name == "data/data_nonfilter_nortar.csv":
            self.message = "data is not processed but the target is normalized"
            self.class_name = "RT"
            self.values_names = {"feature_name", "target"}
            self.class_len = 29
            self.fname = "model_nonfilter_nontrun"

        elif file_name == "data/data_filter_nortar_trun.csv":
            self.message = "data is truncated"
            self.class_name = "RT"
            self.values_names = {"feature_name", "target"}
            self.class_len = 29
            self.fname = "model_filter_trun"
       
        else:
            print("File not found, exiting")
            exit(1)

