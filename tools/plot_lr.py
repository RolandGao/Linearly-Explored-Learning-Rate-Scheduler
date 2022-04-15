from fileinput import filename
import sys
import argparse
import json
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line options (filename and mode)."""
    parser = argparse.ArgumentParser(description="Plot lr vs time (iterations or epoch)")
    help_s = "log file location"
    parser.add_argument("--filename", help=help_s, required=True, type=str)
    help_s, choices = "lr plot mode", ["epoch", "iter"]
    parser.add_argument("--mode", help=help_s, choices=choices, default="epoch", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def plot_les_lr_fun(lrs,start_epoch, max_epoch, filename=None):
    """Visualizes les lr function."""
    epochs = list(range(start_epoch, max_epoch + 1))
    plt.plot(epochs, lrs, ".-")
    plt.title("lr_policy: {}".format(max_epoch))
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.ylim(bottom=0)

    if filename:
        print("saving lr plot")
        plt.savefig(filename + ".png")
        plt.clf()
    else:
        print("showing file")
        plt.show()

    
def extract_lrs(filename, mode):
    lrs = []
    with open(filename) as f:
        start_epoch = -1 if mode == "epoch" else 1
        max_epoch = -1 if mode == "epoch" else 0
        for line in f.readlines():
            if start_epoch == -1 and "Start epoch" in line: 
                start_epoch = int(line.split()[4])

            # if max_epoch == -1 and "MAX_EPOCH" in line: 
            #     line = line.split()[3:]
            #     line = " ".join(line)
            #     line = json.loads(line)
            #     max_epoch = int(line["OPTIM"]["MAX_EPOCH"])

            if mode == "epoch":
                if "train_epoch" in line:
                    line = line.split("{")[1]
                    line = "{" + line
                    # dictionary of data
                    line = json.loads(line)
                    lrs.append(line["lr"])
                    max_epoch = int(line["epoch"].split("/")[0])
            elif mode == "iter":
                if "best_lr" in line:
                    line = line.split()[3:]
                    line = " ".join(line)
                    line = json.loads(line)
                    lrs.append(line["best_lr"])
                    max_epoch += 1
                
    return lrs, start_epoch, max_epoch

def main():
    parser = parse_args()
    filename = parser.filename
    mode = parser.mode
    print(filename)
    print(mode)
    
    lrs, start_epoch, max_epoch = extract_lrs(filename, mode)
    plot_les_lr_fun(lrs, start_epoch, max_epoch, filename)
    
if __name__ == "__main__":
    main()
        