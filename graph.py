from libraries import *
from scipy.interpolate import make_interp_spline


# function to print and save graphs
def graph(acc_list, loss_list, epochs):
    plt.figure()
    # creating an epoch list from total epochs
    epoch_list = np.array([x for x in range(1, epochs+1)])
    loss_list = np.array(loss_list)
    acc_list = np.array(acc_list)
    plt.axis([0, epoch_list.max(), 0, 100])
    plt.plot(epoch_list, acc_list, 'o-')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy vs Epoch")
    plt.savefig("accuracy_graph.png")
    plt.figure()
    plt.axis([0, epoch_list.max(), 0, loss_list.max()])
    plt.plot(epoch_list, loss_list, 'o-')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.savefig("loss_graph.png")
    # plt.show()
    return
