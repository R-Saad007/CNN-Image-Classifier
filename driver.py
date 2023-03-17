from libraries import *
from train import train
from test import test
from graph import graph

# Driver function
def driver():
    # list to store accuracies and loss with epochs
    acc_list = []
    loss_list = []
    # flag to save the model
    save_flag = False
    # Input from user
    epoch = int(input("Enter number of epochs:\t"))
    for x in range(1,epoch+1):
        if x == epoch:
            save_flag = True
        collected_loss = train(x)
        loss_list.append(collected_loss)
        collected_acc = test(save_flag)
        acc_list.append(collected_acc)
    return acc_list,loss_list, epoch

if __name__ == "__main__":
    # list to store accuracies and loss with epochs
    acc_list = []
    loss_list = []
    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    print("Starting Image Classification")
    # whatever you are timing goes here
    acc_list,loss_list,epochs = driver()
    # making accuracy and training loss graphs
    graph(acc_list, loss_list,epochs)
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
