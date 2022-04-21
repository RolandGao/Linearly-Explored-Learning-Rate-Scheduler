#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Plotting functions."""

import colorlover as cl
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as offline
import pycls.core.logging as logging
import torch
import numpy as np

def get_default_colors():
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.arange(255).view(-1, 1) * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def get_plot_colors(max_colors, color_format="pyplot"):
    """Generate colors for plotting."""
    # colors=np.array([[128,0,0],[255,0,0],[255,165,0],[184,134,11],[0,100,0],[30,144,255],[0,0,255],[75,0,130]])
    # return list(colors/255.0)
    colors = cl.scales["11"]["qual"]["Paired"]
    if max_colors > len(colors):
        colors = cl.to_rgb(cl.interp(colors, max_colors))
    if color_format == "pyplot":
        return [[j / 255.0 for j in c] for c in cl.to_numeric(colors)]
    return colors


def prepare_plot_data(log_files, names, metric="top1_err",phases=("train", "test")):
    """Load logs and extract data for plotting error curves."""
    plot_data = []
    for file, name in zip(log_files, names):
        d, data = {}, logging.sort_log_data(logging.load_log_data(file,["les","les_epoch"]))
        for phase in phases:
            x = data[phase + "_epoch"]["epoch_ind"]
            y = data[phase + "_epoch"][metric]
            d["x_" + phase], d["y_" + phase] = x, y
            d[phase + "_label"] = "[{:5.2f}] ".format(min(y) if y else 0) + name
        plot_data.append(d)
    assert len(plot_data) > 0, "No data to plot"
    return plot_data


def plot_error_curves_plotly(log_files, names, filename, metric="top1_err"):
    """Plot error curves using plotly and save to file."""
    plot_data = prepare_plot_data(log_files, names, metric)
    colors = get_plot_colors(len(plot_data), "plotly")
    # Prepare data for plots (3 sets, train duplicated w and w/o legend)
    data = []
    for i, d in enumerate(plot_data):
        s = str(i)
        line_train = {"color": colors[i], "dash": "dashdot", "width": 1.5}
        line_test = {"color": colors[i], "dash": "solid", "width": 1.5}
        data.append(
            go.Scatter(
                x=d["x_train"],
                y=d["y_train"],
                mode="lines",
                name=d["train_label"],
                line=line_train,
                legendgroup=s,
                visible=True,
                showlegend=False,
            )
        )
        data.append(
            go.Scatter(
                x=d["x_test"],
                y=d["y_test"],
                mode="lines",
                name=d["test_label"],
                line=line_test,
                legendgroup=s,
                visible=True,
                showlegend=True,
            )
        )
        data.append(
            go.Scatter(
                x=d["x_train"],
                y=d["y_train"],
                mode="lines",
                name=d["train_label"],
                line=line_train,
                legendgroup=s,
                visible=False,
                showlegend=True,
            )
        )
    # Prepare layout w ability to toggle 'all', 'train', 'test'
    titlefont = {"size": 18, "color": "#7f7f7f"}
    vis = [[True, True, False], [False, False, True], [False, True, False]]
    buttons = zip(["all", "train", "test"], [[{"visible": v}] for v in vis])
    buttons = [{"label": b, "args": v, "method": "update"} for b, v in buttons]
    layout = go.Layout(
        title=metric + " vs. epoch<br>[dash=train, solid=test]",
        xaxis={"title": "epoch", "titlefont": titlefont},
        yaxis={"title": metric, "titlefont": titlefont},
        showlegend=True,
        hoverlabel={"namelength": -1},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.02,
                "xanchor": "left",
                "y": 1.08,
                "yanchor": "top",
            }
        ],
    )
    # Create plotly plot
    offline.plot({"data": data, "layout": layout}, filename=filename)


def plot_error_curves_pyplot(log_files, names, filename=None, metric="top1_err"):
    """Plot error curves using matplotlib.pyplot and save to file."""
    plot_data = prepare_plot_data(log_files, names, metric)
    colors = get_plot_colors(len(names))
    for ind, d in enumerate(plot_data):
        c, lbl = colors[ind], d["test_label"]
        plt.plot(d["x_train"], d["y_train"], "--", c=c, alpha=0.8)
        plt.plot(d["x_test"], d["y_test"], "-", c=c, alpha=0.8, label=lbl)
    plt.title(metric + " vs. epoch\n[dash=train, solid=test]", fontsize=14)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.grid(alpha=0.4)
    plt.legend()
    if filename:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()

def plot_loss_curves_pyplot(log_files, names, filename=None, metric="top1_err"):
    """Plot error curves using matplotlib.pyplot and save to file."""
    plot_data = prepare_plot_data(log_files, names, metric,("train",))
    colors = get_plot_colors(len(names))
    for ind, d in enumerate(plot_data):
        c= colors[ind]
        lbl=d["train_label"]
        plt.plot(d["x_train"], d["y_train"], "-", c=c, alpha=0.8, label=lbl)
        # lbl=d["test_label"]
        # plt.plot(d["x_test"], d["y_test"], "-", c=c, alpha=0.8, label=lbl)
    plt.title("train loss vs. epoch", fontsize=14)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("train loss", fontsize=14)
    plt.grid(alpha=0.4)
    plt.legend()
    if filename:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()

def main():
    import glob
    import os

    lr_scheduler_keywords=["cos","exp","lin","steps","les-v8-single-cooldown"]
    optimizer_keywords=["adam","adamw","cos","les-v8-single-cooldown"]
    def string_in_list(l,s):
        for x in l:
            if x in s:
                return True
        return False
    # filenames2=glob.glob(f"logs2/*.log")
    # labels2=[filename.split("/")[-1] for filename in filenames2]
    # filenames.extend(filenames2)
    # labels.extend(labels2)
    for topic,keywords in [("lr_scheduler",lr_scheduler_keywords),("optimizer",optimizer_keywords)]:
        filenames=glob.glob(f"Final/*/*.log")
        filenames=[filename for filename in filenames if string_in_list(keywords,filename)]
        labels=[filename.split("/")[-2] for filename in filenames]
        plot_error_curves_pyplot(filenames,labels,f"figures/{topic}_top1_err_vs_time.pdf")
        plot_loss_curves_pyplot(filenames,labels,f"figures/{topic}_loss_vs_time.pdf",metric="loss")

if __name__=="__main__":
    main()
