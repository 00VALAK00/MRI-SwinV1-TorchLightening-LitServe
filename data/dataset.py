import torchvision.datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from data import data_dir
from data.build import build_transform, build_dataset
from sklearn.model_selection import train_test_split


class MriDataset(LightningDataModule):
    def __init__(self, data_directory=data_dir, train_split_size: float = .7, batch_size: int = 64):
        super().__init__()
        self.val_data = None
        self.test_data = None
        self.train_data = None
        self.batch_size = batch_size
        self.split_size = train_split_size
        self.data_dir = data_dir
        self.mri_dataset = build_dataset(self.data_dir)
        self.train_transform = build_transform(is_train=True)
        self.val_transform = build_transform(is_train=False)

    @staticmethod
    def split_dataset(data, split_size: float):
        """Stratified split of the dataset into train and validation subsets."""
        labels = [sample[1] for sample in data]  # Extract labels

        # Perform first stratified split for training_data
        train_indices, val_indices = train_test_split(
            range(len(data)),
            train_size=split_size,
            stratify=labels,
            random_state=42
        )
        train_data = Subset(data, train_indices)
        val_data = Subset(data, val_indices)

        return train_data, val_data

    @staticmethod
    def apply_transform(subset, transform):
        """Apply the specified transform to each sample in a Subset."""
        transformed_data = [(transform(sample[0]), sample[1]) for sample in subset]
        return transformed_data

    def setup(self, stage: str):
        self.train_data, self.test_data = self.split_dataset(self.mri_dataset, split_size=self.split_size)

        self.train_data = self.apply_transform(self.train_data, self.train_transform)
        self.test_data = self.apply_transform(self.test_data, self.val_transform)

        self.val_data, self.test_data = self.split_dataset(self.test_data, split_size=.5)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
