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
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25) 
        self.dropout4 = nn.Dropout(0.25) 
        self.fc1 = nn.Linear(16 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 90)
        self.fc3 = nn.Linear(90, 40)
        self.fc4 = nn.Linear(40,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = self.dropout4(x)
        x = self.fc4(x)
        return x

# prints the structure of the CNN    
CNN_net = Net()
CNN_net = CNN_net.to(device)
# print(CNN_net)

# defining the loss and optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(CNN_net.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=2,gamma=0.1)
