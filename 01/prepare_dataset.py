import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)


def get_dataset(dataset_path: str, download: bool = False):
    train = datasets.MNIST(
        root=dataset_path, train=True, download=download, transform=transform
    )
    test = datasets.MNIST(
        root=dataset_path, train=False, download=download, transform=transform
    )
    return train, test


def main():
    dataset_path = "/home/ivan/datasets"

    mnist_train_dataset, mnist_test_dataset = get_dataset(dataset_path, download=True)

    train_loader = DataLoader(mnist_train_dataset)
    (image, label) = next(iter(train_loader))
    print(image.shape, label.shape)


if __name__ == "__main__":
    main()
