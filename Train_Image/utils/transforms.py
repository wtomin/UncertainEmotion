import torchvision.transforms as transforms

def train_transforms(img_size):
    transform_list = [transforms.Resize(img_size),
                      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                    ]
    return transforms.Compose(transform_list)

def test_transforms(img_size):
    transform_list = [transforms.Resize(img_size),
                    transforms.ToTensor(),
                    ]
    return transforms.Compose(transform_list)