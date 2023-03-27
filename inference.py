from cnn import Net
import argparse
from libraries import *
from normalize_load_data import *

# inference class
class inference():

    def __init__(self):
        # list to store predicted labels
        self.pred_labels = []
        # list of actual labels
        self.ground_truth_labels = classes
        # model used
        self.model = self.model_init()
        self.data_transforms = self.transform_infer_imgs()
    
    # transformation on images
    def transform_infer_imgs(self):
        data_transforms = transforms.Compose([
        # conversion to tensors
        transforms.ToTensor(),
        # image resize
        transforms.Resize((32,32)),
        # normalizing tensors
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ((mean of 0.5 for R,G,B) (sd of 0.5 for R,G,B)) (ImageNet standard)
        ])
        return data_transforms

    # model
    def model_init(self):
        # To load saved model
        model = Net()
        model.load_state_dict(torch.load(args.model_path))
        # to set dropout and batch normalization layers to evaluation 
        model.eval()
        model = model.to(device)
        return model

    # inference
    def infer(self,obj):
        # inference dataset
        inference_dataset = torchvision.datasets.ImageFolder('./inference_images', transform=obj.data_transforms)
        dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size = 1, shuffle=False, num_workers=NUM_OF_WORKERS)
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                output = obj.model(inputs)
                output = output.to(device)
                _, predicted = torch.max(output.data, 1)
                obj.pred_labels.append(predicted)
        # printing class assigned to assess model inference
        for x in range(0, len(obj.pred_labels)):
            print(f"Class Assigned to image{x+1}:", obj.ground_truth_labels[obj.pred_labels[x]])
        return
    
    # function to display inference images given by the user
    def view_images(self):
        img_list = sorted(os.listdir(args.img_path))
        # list to store all images
        images = []
        for image in img_list:
            image_path = os.path.join(args.img_path, image)
            images.append(cv.imread(image_path))
            
        fig = plt.figure(figsize=(8, 8))
        rows = 5
        for x in range(len(images)):
            fig.add_subplot(rows, 5, x+1)
            plt.title(f'{x+1}')
            plt.imshow(images[x])
            plt.axis('off')
        plt.savefig('inference_input.png')
        return
    
if __name__ == '__main__':
    # Arguments from CLI
    parser = argparse.ArgumentParser(description = 'Input file paths required.')
    parser.add_argument('-img_path', type = str, dest = 'img_path', required =True)
    parser.add_argument('-model_path', type = str, dest = 'model_path', required = True)
    args = parser.parse_args()
    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    print("Starting Image Inference")
    # whatever you are timing goes here
    infer_imgs = inference()
    infer_imgs.infer(infer_imgs)
    infer_imgs.view_images()
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
