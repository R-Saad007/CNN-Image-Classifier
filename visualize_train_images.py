from libraries import *
from normalize_load_data import train_dataloader, classes, BATCH_SIZE

# function to show images
def visualize_data(img):
    # Unnormalizing images
    img = (img/2) * 0.5
    numpy_img = img.numpy()
    plt.imshow(np.transpose(numpy_img,(1,2,0)))
    plt.show()

# get random images with the iter function of the dataloader
data_iter = iter(train_dataloader)
images, labels = next(data_iter) # gives us first output i.e. 16 images of first batch

# # calling our function to display images
visualize_data(torchvision.utils.make_grid(images))

# printing classes of the images
print(' '.join('%s' % classes[labels[j]] for j in range(BATCH_SIZE)))
