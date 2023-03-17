from libraries import *
from normalize_load_data import test_dataloader
from cnn import *

# Test function
def test(save_flag):
    correct = 0
    accuracy = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = CNN_net(images)
            outputs = outputs.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # accuracy
    accuracy = (correct / total) * 100
    print('Finished Testing')
    print('Accuracy on the 10000 test images: %.2f %%' % accuracy)
    print("-----------------------------------------")
    if save_flag:
        # to save the CNN model
        PATH = './cifar_net.pth'
        torch.save(CNN_net.state_dict(), PATH)
        print('Model Saved')
    return accuracy
