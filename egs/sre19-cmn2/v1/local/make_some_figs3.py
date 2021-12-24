#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt

output_dir = "exp/figs/figs_ft3"


def plot_loss_vs_epochs():

    net_dir0 = "exp/xvector_nnets"
    net_dir1 = "resnet34_zir_e256_arc0.3_do0_adam_lr0.01_b512_amp.v2.ft_1000_6000_sgdcos_lr0.05_b128_amp.v2"
    net_dir2 = ".ft_eaffine_rege_w%s_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc%s_we%s_1000_6000_sgdcos_lr0.01_b128_amp.v2"
    w = ["0.001", "0.01", "0.1", "1", "10"]
    colors = ["b", "--r", "-.g", "m", "--c", "-.k"]
    df = []
    for i in range(len(w)):
        wi = w[i]
        net_dir2i = net_dir2 % (wi, wi, wi)
        net_dir = "%s/%s%s" % (net_dir0, net_dir1, net_dir2i)
        file_path = net_dir + "/train.log"
        df_i = pd.read_csv(file_path)
        df.append(df_i)

    cols_h = ["reg-h-enc-0", "reg-h-enc-1", "reg-h-enc-2", "reg-h-enc-3", "reg-h-enc-4"]
    col_e = "reg-h-classif-0"
    col_cxe = "loss-classif"
    col_val_cxe = "val_loss"

    plt.figure()
    for i in range(len(df)):
        df_i = df[i]
        m1 = df_i[cols_h].mean(axis=1) + df_i[col_e]
        x = df_i["epoch"].values.astype(np.int)
        y = m1.values
        kk = x <= 33
        x = x[kk]
        y = y[kk]
        plt.plot(x, y, colors[i], label="w=%s" % (w[i]))

    plt.ylabel("L1 regularization loss")
    plt.xlabel("num. epochs")
    plt.grid()
    plt.legend()
    plt.savefig(output_dir + "/lreg_vs_epochs.pdf")
    plt.close()

    colors = ["b", "r", "g", "m", "c", "k"]
    plt.figure()
    for i in range(len(df)):
        df_i = df[i]

        x = df_i["epoch"].values.astype(np.int)
        y = df_i[col_cxe].values
        y_val = df_i[col_val_cxe].values
        kk = x <= 33
        x = x[kk]
        y = y[kk]
        y_val = y_val[kk]
        plt.plot(x, y, colors[i], label="train-cxe w=%s" % (w[i]))
        plt.plot(x, y_val, "--" + colors[i], label="val-cxe w=%s" % (w[i]))

    plt.ylabel("cross-entropy loss")
    plt.xlabel("num. epochs")
    plt.grid()
    plt.legend()
    plt.savefig(output_dir + "/cxe_vs_epochs.pdf")
    plt.close()

    plt.figure()
    for i in range(len(df)):
        df_i = df[i]

        x = df_i["epoch"].values.astype(np.int)
        y = df_i["acc"].values
        y_val = df_i["val_acc"].values
        kk = x <= 33
        x = x[kk]
        y = y[kk] * 100
        y_val = y_val[kk] * 100
        plt.plot(x, y, colors[i], label="train-acc w=%s" % (w[i]))
        plt.plot(x, y_val, "--" + colors[i], label="val-acc w=%s" % (w[i]))

    plt.ylabel("Accuracy (%)")
    plt.xlabel("num. epochs")
    plt.grid()
    plt.legend()
    plt.savefig(output_dir + "/acc_vs_epochs.pdf")
    plt.close()

    colors = ["b", "--r", "-.g", "m", "--c", "-.k"]

    plt.figure()
    for i in range(len(df)):
        df_i = df[i]

        x = df_i["epoch"].values.astype(np.int)
        y = df_i[col_cxe].values
        kk = x <= 33
        x = x[kk]
        y = y[kk]
        plt.plot(x, y, colors[i], label="w=%s" % (w[i]))

    plt.ylabel("train cross-entropy loss")
    plt.xlabel("num. epochs")
    plt.grid()
    plt.legend()
    plt.savefig(output_dir + "/train_cxe_vs_epochs.pdf")
    plt.close()

    plt.figure()
    for i in range(len(df)):
        df_i = df[i]

        x = df_i["epoch"].values.astype(np.int)
        y = df_i[col_val_cxe].values
        kk = x <= 33
        x = x[kk]
        y = y[kk]
        plt.plot(x, y, colors[i], label="w=%s" % (w[i]))

    plt.ylabel("val. cross-entropy loss")
    plt.xlabel("num. epochs")
    plt.grid()
    plt.legend()
    plt.savefig(output_dir + "/val_cxe_vs_epochs.pdf")
    plt.close()

    plt.figure()
    for i in range(len(df)):
        df_i = df[i]
        x = df_i["epoch"].values.astype(np.int)
        y = df_i["acc"].values
        kk = x <= 33
        x = x[kk]
        y = y[kk] * 100
        plt.plot(x, y, colors[i], label="w=%s" % (w[i]))

    plt.ylabel("train accuracy (%)")
    plt.xlabel("num. epochs")
    plt.grid()
    plt.legend()
    plt.savefig(output_dir + "/train_acc_vs_epochs.pdf")
    plt.close()

    plt.figure()
    for i in range(len(df)):
        df_i = df[i]
        x = df_i["epoch"].values.astype(np.int)
        y = df_i["val_acc"].values
        kk = x <= 33
        x = x[kk]
        y = y[kk] * 100
        plt.plot(x, y, colors[i], label="w=%s" % (w[i]))

    plt.ylabel("val accuracy (%)")
    plt.xlabel("num. epochs")
    plt.grid()
    plt.legend()
    plt.savefig(output_dir + "/val_acc_vs_epochs.pdf")
    plt.close()


def plot_perf_vs_iter_w():

    w = ["0.001", "0.01", "0.1", "1", "10"]
    be = ["be1", "be2", "be3", "be1-snorm", "be2-snorm", "be3-snorm"]
    colors = ["b", "--r", "-.g", "m", "--c", "-.k"]
    dbs = ["sre18", "sre19p", "sre19e"]
    titles = ["SRE18 Eval40%", "SRE19-Prog", "SRE19-Eval"]

    for i in range(len(be)):

        df = []
        for j in range(len(w)):
            in_file = "%s/table1_w%s_%s.csv" % (output_dir, w[j], be[i])
            df.append(pd.read_csv(in_file, index_col=False))

        xlabels = df[0]["system"].values
        locs = [l for l in range(len(xlabels))]
        for k in range(len(dbs)):
            title_k = titles[k]
            dbk = dbs[k]
            plt.figure()

            for j in range(len(w)):
                df_j = df[j]

                y = df_j[dbk + "-eer"].values
                plt.plot(y, colors[j], label="w=%s" % (w[j]))

            plt.ylabel("EER(%)")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_%s_eer_vs_epochs.pdf" % (output_dir, dbk, be[i]))
            plt.close()

            plt.figure()
            for j in range(len(w)):
                df_j = df[j]

                y = df_j[dbk + "-min-dcf"]
                plt.plot(y, colors[j], label="w=%s" % (w[j]))

            plt.ylabel("MinCprimary")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_%s_mindcf_vs_epochs.pdf" % (output_dir, dbk, be[i]))
            plt.close()

            plt.figure()
            for j in range(len(w)):
                df_j = df[j]
                y = df_j[dbk + "-act-dcf"]
                plt.plot(y, colors[j], label="w=%s" % (w[j]))

            plt.ylabel("ActCprimary")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_%s_actdcf_vs_epochs.pdf" % (output_dir, dbk, be[i]))
            plt.close()


def plot_perf_vs_iter_nnet():

    nnet_nb = ["0", "1", "2"]
    nnet_name = ["ResNet34", "SE-ResNet34", "TSE-ResNet34"]
    be = ["be1", "be2", "be3", "be1-snorm", "be2-snorm", "be3-snorm"]
    colors = ["b", "--r", "-.g", "m", "--c", "-.k"]
    dbs = ["sre18", "sre19p", "sre19e"]
    titles = ["SRE18 Eval40%", "SRE19-Prog", "SRE19-Eval"]

    for i in range(len(be)):

        df = []
        for j in range(len(nnet_nb)):
            in_file = "%s/table1_nnet%s_%s.csv" % (output_dir, nnet_nb[j], be[i])
            df.append(pd.read_csv(in_file, index_col=False))

        xlabels = df[0]["system"].values
        locs = [l for l in range(len(xlabels))]
        for k in range(len(dbs)):
            title_k = titles[k]
            dbk = dbs[k]
            plt.figure()

            for j in range(len(nnet_nb)):
                df_j = df[j]

                y = df_j[dbk + "-eer"].values
                plt.plot(y, colors[j], label="%s" % (nnet_name[j]))

            plt.ylabel("EER(%)")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_nnets_%s_eer_vs_epochs.pdf" % (output_dir, dbk, be[i]))
            plt.close()

            plt.figure()
            for j in range(len(nnet_nb)):
                df_j = df[j]

                y = df_j[dbk + "-min-dcf"]
                plt.plot(y, colors[j], label="%s" % (nnet_name[j]))

            plt.ylabel("MinCprimary")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig(
                "%s/%s_nnets_%s_mindcf_vs_epochs.pdf" % (output_dir, dbk, be[i])
            )
            plt.close()

            plt.figure()
            for j in range(len(nnet_nb)):
                df_j = df[j]
                y = df_j[dbk + "-act-dcf"]
                plt.plot(y, colors[j], label="%s" % (nnet_name[j]))

            plt.ylabel("ActCprimary")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig(
                "%s/%s_nnets_%s_actdcf_vs_epochs.pdf" % (output_dir, dbk, be[i])
            )
            plt.close()


def plot_perf_vs_iter_be1():

    w = ["0.001", "0.01", "0.1", "1", "10"]
    w = ["1"]
    be = ["be1-snorm", "be2-snorm", "be3-snorm"]
    colors = ["b", "--r", "-.g", "m", "--c", "-.k"]
    dbs = ["sre18", "sre19p", "sre19e"]
    titles = ["SRE18 Eval40%", "SRE19-Prog", "SRE19-Eval"]

    for j in range(len(w)):

        df = []
        for i in range(len(be)):
            in_file = "%s/table1_w%s_%s.csv" % (output_dir, w[j], be[i])
            df.append(pd.read_csv(in_file, index_col=False))

        xlabels = df[0]["system"].values
        locs = [l for l in range(len(xlabels))]
        for k in range(len(dbs)):
            title_k = titles[k]
            dbk = dbs[k]
            plt.figure()

            for i in range(len(be)):
                df_i = df[i]

                y = df_i[dbk + "-eer"]
                plt.plot(y, colors[i], label=be[i])

            plt.ylabel("EER(%)")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_w%s_eer_vs_epochs.pdf" % (output_dir, dbk, w[j]))
            plt.close()

            plt.figure()
            for i in range(len(be)):
                df_i = df[i]

                y = df_i[dbk + "-min-dcf"]
                plt.plot(y, colors[i], label=be[i])

            plt.ylabel("MinCprimary")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_w%s_mindcf_vs_epochs.pdf" % (output_dir, dbk, w[j]))
            plt.close()

            plt.figure()
            for i in range(len(be)):
                df_i = df[i]
                y = df_i[dbk + "-act-dcf"]
                plt.plot(y, colors[i], label=be[i])

            plt.ylabel("ActCprimary")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_w%s_actdcf_vs_epochs.pdf" % (output_dir, dbk, w[j]))
            plt.close()


def plot_perf_vs_iter_be2():

    w = ["0.001", "0.01", "0.1", "1", "10"]
    w = ["1"]
    be = ["be1", "be2", "be3", "be1-snorm", "be2-snorm", "be3-snorm"]
    colors = ["b", "r", "g", "--b", "--r", "--g"]
    dbs = ["sre18", "sre19p", "sre19e"]
    titles = ["SRE18 Eval40%", "SRE19-Prog", "SRE19-Eval"]

    for j in range(len(w)):

        df = []
        for i in range(len(be)):
            in_file = "%s/table1_w%s_%s.csv" % (output_dir, w[j], be[i])
            df.append(pd.read_csv(in_file, index_col=False))

        xlabels = df[0]["system"].values
        locs = [l for l in range(len(xlabels))]
        for k in range(len(dbs)):
            title_k = titles[k]
            dbk = dbs[k]
            plt.figure()

            for i in range(len(be)):
                df_i = df[i]

                y = df_i[dbk + "-eer"]
                plt.plot(y, colors[i], label=be[i])

            plt.ylabel("EER(%)")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_w%s_eer_vs_epochs.pdf" % (output_dir, dbk, w[j]))
            plt.close()

            plt.figure()
            for i in range(len(be)):
                df_i = df[i]

                y = df_i[dbk + "-min-dcf"]
                plt.plot(y, colors[i], label=be[i])

            plt.ylabel("MinCprimary")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_w%s_mindcf_vs_epochs.pdf" % (output_dir, dbk, w[j]))
            plt.close()

            plt.figure()
            for i in range(len(be)):
                df_i = df[i]
                y = df_i[dbk + "-act-dcf"]
                plt.plot(y, colors[i], label=be[i])

            plt.ylabel("ActCprimary")
            plt.xlabel("model")
            plt.xlim(0, len(y) - 1)
            plt.xticks(locs, xlabels, rotation=10, fontsize=8)
            plt.grid()
            plt.legend()
            plt.title(title_k)
            plt.tight_layout()
            plt.savefig("%s/%s_w%s_actdcf_vs_epochs.pdf" % (output_dir, dbk, w[j]))
            plt.close()


if __name__ == "__main__":

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # plot_loss_vs_epochs()
    # plot_perf_vs_iter_w()
    # plot_perf_vs_iter_be2()
    plot_perf_vs_iter_nnet()
