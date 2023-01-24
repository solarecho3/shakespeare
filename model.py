import torch
import torchvision

print(f'Running {__name__}from {__file__}')

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

training_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# each element in the dataloader iterable will
# return a batch of <batch_size> features and labels.
batch_size = 64

train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

print(torch.randint(10, size=(1,)))