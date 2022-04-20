import sys
import argparse
import json
import matplotlib.pyplot as plt
import os
import glob

def parse_args():
    """Parse command line options (filename and mode)."""
    parser = argparse.ArgumentParser(description="Plot lr vs time (iterations or epoch)")
    help_s = "log file location"
    parser.add_argument("--filename", help=help_s, required=True, type=str)
    help_s, choices = "lr plot mode", ["epoch", "iter"]
    parser.add_argument("--mode", help=help_s, choices=choices, default="iter", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

# def plot_les_helper(x,y,legend,)

def plot_les_fun(data, start_epoch, max_epoch, mode, label, filename=None):

    """Visualizes les lr function."""
    epochs = list(range(start_epoch, max_epoch + 1))
    plt.plot(epochs, data, ".-")
    plt.title(f"lr_policy: les - {max_epoch}")
    if mode == "epoch":
        plt.xlabel("epochs")
    elif mode == "iter":
        plt.xlabel("iterations")
    plt.ylabel(label)
    plt.ylim(bottom=0)

    if filename:
        print("saving lr plot")
        plt.savefig(filename + ".png")
        plt.clf()
    else:
        print("showing file")
        plt.show()
def plot_lrs(filenames,labels,out_dir):
    for filename,label in zip(filenames,labels):
        lrs, weights_first, weights_last, start_epoch, max_epoch=extract_les_data(filename,"iter")
        iterations=list(range(start_epoch, max_epoch + 1))
        if len(lrs)>0:
            plt.plot(iterations,lrs,label=label)
    plt.xlabel("iterations")
    plt.ylabel("lr")
    plt.legend()
    os.makedirs(out_dir,exist_ok=True)
    plt.savefig(os.path.join(out_dir,"lrs.pdf"))
    plt.show()

def plot_weight_norms(filenames,labels,out_dir):
    for filename,label in zip(filenames,labels):
        lrs, weights_first, weights_last, start_epoch, max_epoch=extract_les_data(filename,"iter")
        iterations=list(range(start_epoch, max_epoch + 1))
        if len(weights_first)>0:
            plt.plot(iterations,weights_first,label=f"w1_{label}")
        if len(weights_last)>0:
            plt.plot(iterations,weights_last,label=f"w2_{label}")
    plt.xlabel("iterations")
    plt.ylabel("L2 norm")
    plt.legend()
    os.makedirs(out_dir,exist_ok=True)
    plt.savefig(os.path.join(out_dir,"weight_norms.pdf"))
    plt.show()


def extract_les_data(filename, mode):
    lrs = []
    weights_first = []
    weights_last = []
    with open(filename) as f:
        start_epoch = -1 if mode == "epoch" else 1
        max_epoch = -1 if mode == "epoch" else 0
        for line in f.readlines():
            if start_epoch == -1 and "Start epoch" in line:
                start_epoch = int(line.split()[4])

            if mode == "epoch":
                is_train_epoch = "train_epoch" in line
                is_les_epoch = "les_epoch" in line
                if is_les_epoch or is_train_epoch:
                    line = line.split("{")[1]
                    line = "{" + line
                    # dictionary of data
                    line = json.loads(line)

                    if is_train_epoch:
                        lrs.append(line["lr"])
                        max_epoch = int(line["epoch"].split("/")[0])

                    if is_les_epoch:
                        if "weight_norm_first_layer" in line:
                            weights_first.append(line["weight_norm_first_layer"])
                            weights_last.append(line["weight_norm_last_layer"])
                        max_epoch = int(line["epoch"])

            elif mode == "iter":
                if "best_lr" in line:
                    line = line.split()[3:]
                    line = " ".join(line)
                    line = json.loads(line)
                    lrs.append(line["best_lr"])
                    if "weight_norm_first_layer" in line:
                        weights_first.append(line["weight_norm_first_layer"])
                        weights_last.append(line["weight_norm_last_layer"])
                    max_epoch += 1

    return lrs, weights_first, weights_last, start_epoch, max_epoch

def main():
    parser = parse_args()
    filename = parser.filename
    mode = parser.mode
    print(filename)
    print(mode)

    lrs, weights_first, weights_last, start_epoch, max_epoch = extract_les_data(filename, mode)
    plot_les_fun(lrs, start_epoch, max_epoch, mode, label="learning rate",filename=filename + "_lr")

    plot_les_fun(weights_first, start_epoch, max_epoch, mode, label="last weight norm",filename=filename + "_wn_last")
    plot_les_fun(weights_last, start_epoch, max_epoch, mode, label="first weight norm",filename=filename + "_wn_first")


def main2():
    filenames=glob.glob("Final/*/*.log")
    labels=[os.path.normpath(filename).split(os.sep)[-2] for filename in filenames]
    # filenames.extend(["logs/version8.log"])
    # labels.extend(["version8"])
    plot_lrs(filenames,labels,"figures")
if __name__ == "__main__":
    # main()
    main2()
    # plot_weight_norms("../Final","figures")
