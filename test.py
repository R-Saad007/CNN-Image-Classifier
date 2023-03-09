from libraries import *
from normalize_load_data import test_dataloader
from cnn import *

# Test function
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = CNN_net(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # accuracy
    print('Finished Testing')
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
    # to save the CNN model
    PATH = './cifar_net.pth'
    torch.save(CNN_net.state_dict(), PATH)
    print('Model Saved')