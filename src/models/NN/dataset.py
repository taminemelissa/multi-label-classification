import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils.tools import concatenate_features


class TrackDataset(Dataset):
    def __init__(self, df, train, test):
        self.df = df
        self.train = train
        self.test = test
        self.all_audio_features = self.df[:]['audio_features']
        self.all_usage_features = self.df[:]['usage_features']
        self.all_features = concatenate_features(self.df)
        self.all_labels = np.array(
            self.df.drop(['song_index', 'audio_features', 'usage_features', 'artist_name', 'song_title'], axis=1))
        self.train_ratio = int(0.85 * len(self.df))
        self.valid_ratio = len(self.df) - self.train_ratio

        # set the training data features and labels
        if self.train:
            print(f"Number of training features: {self.train_ratio}")
            self.audio_features = list(self.all_audio_features[:self.train_ratio])
            self.usage_features = list(self.all_usage_features[:self.train_ratio])
            self.features = list(self.all_features[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])

        # set the validation features and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation features: {self.valid_ratio}")
            self.audio_features = list(self.all_audio_features[-self.valid_ratio:-100])
            self.usage_features = list(self.all_usage_features[-self.valid_ratio:-100])
            self.features = list(self.all_features[-self.valid_ratio:-100])
            self.labels = list(self.all_labels[-self.valid_ratio:-100])

        # set the test features and labels, only last 100 tracks
        elif self.test == True and self.train == False:
            self.audio_features = list(self.all_audio_features[-100:])
            self.usage_features = list(self.all_usage_features[-100:])
            self.features = list(self.all_features[-100:])
            self.labels = list(self.all_labels[-100:])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        audio_feature = self.audio_features[index]
        usage_feature = self.usage_features[index]
        feature = self.features[index]
        targets = self.labels[index]

        return {
            'audio_feature': torch.tensor(audio_feature, dtype=torch.float32),
            'usage_feature': torch.tensor(usage_feature, dtype=torch.float32),
            'feature': torch.tensor(feature, dtype=torch.float32)
            'label': torch.tensor(targets, dtype=torch.float32)
        }
