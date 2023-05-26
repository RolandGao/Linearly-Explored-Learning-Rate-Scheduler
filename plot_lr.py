import sys
import argparse
import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

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
def compute_epoch_means(x,epochs=50):
    return np.array(x).reshape((epochs,-1)).mean(axis=1)
def plot_lrs(filenames,labels,out_dir):
    epochs=50
    for filename,label in zip(filenames,labels):
        lrs, weights_first, weights_last, start_epoch, max_epoch=extract_les_data(filename,"iter")
        if len(lrs)>0:
            # lrs=np.convolve(lrs,np.ones(50),"valid")
            lrs=compute_epoch_means(lrs,epochs)
            iterations=list(range(len(lrs)))
            plt.plot(iterations,lrs,label=label)
    plt.xlabel("iterations")
    plt.ylabel("lr")
    plt.legend()
    os.makedirs(out_dir,exist_ok=True)
    plt.savefig(os.path.join(out_dir,"lrs.pdf"))
    plt.show()

def plot_weight_norms(filenames,labels,out_dir):
    epochs=50
    for filename,label in zip(filenames,labels):
        lrs, weights_first, weights_last, start_epoch, max_epoch=extract_les_data(filename,"iter")
        if len(weights_first)>0:
            # weights_first=compute_epoch_means(weights_first,epochs)
            iterations=list(range(len(weights_first)))
            plt.plot(iterations,weights_first,label=f"{label}")
        # if len(weights_last)>0:
        #     print(label)
        #     print(len(weights_last))
        #     plt.plot(iterations,weights_last,label=f"w2_{label}")
    plt.xlabel("iterations")
    plt.ylabel("L2 norm")
    plt.legend()
    os.makedirs(out_dir,exist_ok=True)
    plt.savefig(os.path.join(out_dir,"weight_norms.pdf"))
    plt.show()


def extract_les_data(filename):
    lrs = []
    weights_first = []
    weights_last = []
    losses=[]
    loss_diffs=[]
    with open(filename) as f:
        for line in f.readlines():
            if "best_lr" in line:
                line = line.split()[3:]
                line = " ".join(line)
                line = json.loads(line)
                lr=line["best_lr"]
                loss=line["loss"]
                lr_to_loss=line["lr_to_loss"]
                lr_to_loss={round(float(lr),4):loss for lr,loss in lr_to_loss.items()}
                loss_diff=loss-lr_to_loss[lr]
                lrs.append(lr)
                losses.append(loss)
                loss_diffs.append(loss_diff)
                if "weight_norm_first_layer" in line:
                    weights_first.append(line["weight_norm_first_layer"])
                    weights_last.append(line["weight_norm_last_layer"])

    return lrs, weights_first, weights_last,losses,loss_diffs

def second_loader_main():
    # filenames=glob.glob(f"Final/*/*.log")
    filenames=[
        "Final/les-v7/stdout-4.log",
        "Final/les-v8/stdout-4.log",
        "Final/les-v8-secondloader/stdout-4.log",
    ]
    labels=[
        "cos",
        "one batch",
        "two independent batches"
    ]
    plot_lrs(filenames, labels, "tools/figures")
    # plot_weight_norms(filenames,labels,"figures")
def weightnorm_main():
    # filenames=glob.glob(f"Final/*/*.log")
    filenames=[
        "Final/les-v7/stdout-4.log",
        "Final/les-v2/stdout-v2.log",
        "Final/les-v8/stdout-4.log",
        "Final/les-v10/stdout-4.log"
    ]
    labels=[
        "cos",
        "without linear warm up",
        "with linear warm up",
        "v10"
    ]
    # plot_lrs(filenames,labels,"figures")
    plot_weight_norms(filenames, labels, "tools/figures")
def plot_lrs_helper(lrs,epochs,out_dir):
    if len(lrs)>0:
        # lrs=np.convolve(lrs,np.ones(50),"valid")
        lrs=compute_epoch_means(lrs,epochs)
        iterations=list(range(len(lrs)))
        plt.plot(iterations,lrs)
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.legend()
    os.makedirs(out_dir,exist_ok=True)
    plt.savefig(os.path.join(out_dir,"lrs.pdf"))
    plt.show()
def plot_les_fun(data, label, filename=None):
    plt.plot(list(range(len(data))), data, "-")
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
def subinterval(v,p=0.1):
    l=[]
    for x in v:
        l.append(x[:int(len(x)*p)])
    return l
def smooth_data(v,):
    l = []
    y=np.ones(100)/100
    for x in v:
        if len(x)>0:
            l.append(np.convolve(x,y,"same"))
        else:
            l.append(x)
    return l
def main():
    print(list(os.listdir(".")))
    out_dir="v9_1000epochs"
    filename="logs/version9.log"

    # filename
    lrs, weights_first, weights_last,losses,loss_diffs = extract_les_data(filename)
    # plot_lrs_helper(lrs,1000,"v9_1000epochs")
    lrs, weights_first, weights_last,losses,loss_diffs=\
        subinterval([lrs, weights_first, weights_last,losses,loss_diffs],p=0.01)
    # lrs, weights_first, weights_last, losses, loss_diffs=smooth_data([lrs, weights_first, weights_last,losses,loss_diffs])
    plt.plot(lrs,label="lrs")
    plt.plot(losses,label="losses")
    plt.plot(loss_diffs, label="losses_diffs")
    plt.plot(weights_first, label="weight_first")
    plt.xlabel("iterations")
    plt.ylabel("values")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "summary.pdf"))
    plt.show()
    # plot_les_fun(losses, label="losses",filename=filename + "_loss")
    # plot_les_fun(loss_diffs, label="losses", filename=filename + "_lossdiff")
    # plot_les_fun(weights_first, label="last weight norm",filename=filename + "_wn_last")
    # plot_les_fun(weights_last, label="first weight norm",filename=filename + "_wn_first")
if __name__ == "__main__":
    main()
    # weightnorm_main()
    # second_loader_main()
