import fastbook
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import tensor
from torch.utils.data import DataLoader


def one_hot(digit):
    return tensor([float(i == digit) for i in range(10)]).unsqueeze(1)


pic_to_matrix = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
matrix_to_column = transforms.Compose([
    transforms.Lambda(torch.flatten),
    transforms.Lambda(lambda x: x.unsqueeze(1)),
    transforms.Lambda(lambda x: x / 2 + 0.5),
])
pic_to_column = transforms.Compose([
    pic_to_matrix,
    matrix_to_column,
])


def to_pic(pixels):
    digit = pixels.numpy().reshape(28,28)
    digit = digit.astype(np.uint8)
    return Image.fromarray(digit).resize((100, 100))


path = fastbook.untar_data(fastbook.URLs.MNIST)


def pic_loader(pic_transform, label_transform=None):
    def loader(path):
        return torchvision.datasets.ImageFolder(
            path.as_posix(),
            transform=pic_transform,
            target_transform=label_transform, 
        )
    return loader


def load_mnist(loader, train_proportion=0.8):
    full_dataset = loader(path / "training")
    
    
    train_size = int(train_proportion * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    training_set, validation_set = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

    # Dataset using the "testing" folder
    testing_set = loader(path / "testing")
    
    return training_set, testing_set, validation_set


def datasets(train_proportion=0.8):
    loader = pic_loader(pic_transform=pic_to_column, label_transform=one_hot)
    training_set, testing_set, validation_set = load_mnist(loader, train_proportion=0.8)

    print(len(training_set), len(testing_set), len(validation_set))
    
    training_set = DataLoader(training_set, batch_size=30, shuffle=True)
    testing_set = DataLoader(testing_set, batch_size=30, shuffle=True)
    validation_set = DataLoader(validation_set, batch_size=30, shuffle=True)
        
    return training_set, testing_set, validation_set


def sample(n, batch_size=10):
    loader = pic_loader(pic_transform=pic_to_column, label_transform=one_hot)
    full_dataset = loader(path / "training")
    
    data_set, _ = torch.utils.data.random_split(full_dataset, [n, len(full_dataset) - n])
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)
