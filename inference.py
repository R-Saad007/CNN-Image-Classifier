from libraries import *
import os
from normalize_load_data import *
from cnn import Net

IMG_PATH = './inference_images/inf_imgs/'
MODEL_PATH = './cifar_net.pth'
# inference function
def inference():
    # list to store predicted labels
    pred_labels = []
    # list of actual labels
    ground_truth_labels = classes
    # To load saved model
    CNN = Net()
    CNN.load_state_dict(torch.load(MODEL_PATH))
    # to set dropout and batch normalization layers to evaluation 
    CNN.eval()
    CNN = CNN.to(device)
    # transformation on images
    data_transforms = transforms.Compose([
    # conversion to tensors
    transforms.ToTensor(),
    # image resize
    transforms.Resize((32,32)),
    # normalizing tensors
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # ((mean of 0.5 for R,G,B) (sd of 0.5 for R,G,G))
    ])
    # inference dataset
    inference_dataset = torchvision.datasets.ImageFolder('./inference_images', transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size = 1, shuffle=False, num_workers=NUM_OF_WORKERS)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            output = CNN(inputs)
            output = output.to(device)
            _, predicted = torch.max(output.data, 1)
            pred_labels.append(predicted)
    # printing class assigned to assess model inference
    for x in range(0, len(pred_labels)):
        print(f"Class Assigned to image{x+1}:", ground_truth_labels[pred_labels[x]])
    return

# function to display inference images given by the user
def view_images():
    img_list = sorted(os.listdir(IMG_PATH))
    # list to store all images
    images = []
    for image in img_list:
        image_path = os.path.join(IMG_PATH, image)
        images.append(cv.imread(image_path))
           
    for x in images:
        plt.figure()
        plt.imshow(x)
    plt.show()
    return

if __name__ == '__main__':
    # For calculating execution time
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    print("Starting Image Inference")
    # whatever you are timing goes here
    inference()
    view_images()
    # end.record()
    # Waits for everything to finish running
    # torch.cuda.synchronize()
    # print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
