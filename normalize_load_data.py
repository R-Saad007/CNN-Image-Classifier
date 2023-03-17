from libraries import *
torch.manual_seed(7)


# python image matrices have a range of [0,1], hence we normalize them to [-1,1] tensors
# transform function
transform = transforms.Compose([
    # conversion to tensors
    transforms.ToTensor(),
    transforms.Resize((32,32)),
    # normalizing tensors
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ((mean of 0.5 for R,G,B) (sd of 0.5 for R,G,G)) (ImageNet standard)
])

# transform function for data augmentation
transform_augmented = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.CenterCrop(10),
    transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
    transforms.Resize((32,32))
])

# setting batch size (multiples of 2 i.e. 2,4,8,16,32)
BATCH_SIZE = 4

# setting number of workers
NUM_OF_WORKERS= 2

# loading train and test data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# augmented dataset
augmented_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
augmented_dataset = torch.utils.data.ConcatDataset([augmented_dataset, train_data])

# creating dataloaders
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True, num_workers=NUM_OF_WORKERS)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_OF_WORKERS)
data_augment_loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=NUM_OF_WORKERS)

# classes to avoid any duplicates
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
