from libraries import *
from normalize_load_data import train_dataloader, test_dataloader
from train import train
from test import test

# Driver function
def driver():
    train()
    test()

if __name__ == "__main__":
    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    print("Starting Image Classification")
    # whatever you are timing goes here
    driver()
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds