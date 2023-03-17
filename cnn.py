from libraries import *
from torch.optim.lr_scheduler import StepLR

# CNN class
class Net(nn.Module):
# add more layers to check results, filter size, neuron number,
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16,5000,5)
        # self.conv4 = nn.Conv2d(400,2000,5)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.2) 
        self.fc1 = nn.Linear(200 * 5 * 5, 220)
        self.fc2 = nn.Linear(220, 90)
        self.fc3 = nn.Linear(90, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.dropout1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

CNN_net = Net()
CNN_net = CNN_net.to(device)

# defining the loss and optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(CNN_net.parameters(), lr=0.001, momentum=0.9)
# scheduler for dynamically adjusting the learning rate
scheduler = StepLR(optimizer, step_size=2,gamma=0.1)
