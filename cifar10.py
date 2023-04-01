from torchvision.datasets import CIFAR10 as DATA
from torch.utils.data import DataLoader
from torchvision import transforms

class CIFAR10():
    def __init__(self, train_batch_size: int = 512, test_batch_size: int = 100):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def transform_train(self):
        return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    def transform_test(self):
        return transforms.Compose([
        transforms.ToTensor(),
        ])

    def train_data(self):
        train_dataset = DATA('./data/', train=True, download=True, transform=self.transform_train())
        return DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def test_data(self):
        test_dataset = DATA('./data/', train=False, download=True, transform=self.transform_test())
        return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)
