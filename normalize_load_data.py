from libraries import *

# python image matrices have a range of [0,1], hence we normalize them to [-1,1] tensors
# transform function
transform = transforms.Compose([
    # conversion to tensors
    transforms.ToTensor(),
    # normalizing tensors
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # ((mean of 0.5 for R,G,B) (sd of 0.5 for R,G,B))
])

# setting batch size
BATCH_SIZE = 5

# setting number of workers
NUM_OF_WORKERS = 2

# loading train and test data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# creating dataloaders
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True, num_workers=NUM_OF_WORKERS)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_OF_WORKERS)

# classes to avoid any duplicates
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
