from torchvision import transforms
from torchvision.datasets import ImageFolder


def build_dataset(data_dir,):

    return ImageFolder(root=data_dir, allow_empty=False)


def build_transform(is_train: bool):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandAugment(num_ops=2,
                                   interpolation=transforms.InterpolationMode.BILINEAR),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


