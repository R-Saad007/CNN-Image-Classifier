from libraries import *
from normalize_load_data import train_dataloader
# we can use the dataloader below for data augmentation
# from normalize_load_data import data_augment_loader
from cnn import *

# Train function
def train(epochs):
    print("Mini-Batches of 2000 images will be used")
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # making use of GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = CNN_net.forward(inputs)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            loss = loss.to(device)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += running_loss
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('Epoch: %d | Mini-Batch: %5d | loss: %.3f' % (epochs, i + 1, running_loss / 2000))
                running_loss = 0.0
        print("-----------------------------------------")
        # Calling scheduler to adjust learning rate
        scheduler.step()
    print('Finished Training')
    return epoch_loss
