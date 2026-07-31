#!/usr/bin/env python
"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)  
"""
import os

import numpy as np
import pandas as pd

from hyperion.hyp_defs import float_cpu, config_logger
from hyperion.np.metrics.verification_evaluator import (
    VerificationAdvAttackEvaluator as Eval,
)

filenames = [
    "voxceleb1_attack_tar_snr_results.csv",
    "voxceleb1_attack_non_snr_results.csv",
    "voxceleb1_attack_tar_linf_results.csv",
    "voxceleb1_attack_non_linf_results.csv",
]

output_dir = "exp/figs/resnet34_1/"
base_res_dir = "exp/scores/"


def plot_figs1(
    res_dirs1,
    legends,
    title_base,
    fig_base,
    fmt=["b", "r", "g", "m", "c", "y"],
    clean_ref=0,
):
    df = []
    for i in range(len(res_dirs1)):
        file_path = "%s/%s/%s" % (base_res_dir, res_dirs1[i], filenames[0])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + "_tar_snr"
    Eval.plot_dcf_eer_vs_stat_v2(
        df,
        "snr",
        fig_file,
        clean_ref=clean_ref,
        xlabel="SNR(dB)",
        higher_better=True,
        legends=legends,
        fmt=fmt,
        title=title_base + " attacks on target trials",
        font_size=13,
    )
    df = []
    for i in range(len(res_dirs1)):
        file_path = "%s/%s/%s" % (base_res_dir, res_dirs1[i], filenames[1])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + "_non_snr"
    Eval.plot_dcf_eer_vs_stat_v2(
        df,
        "snr",
        fig_file,
        clean_ref=clean_ref,
        xlabel="SNR(dB)",
        higher_better=True,
        legends=legends,
        fmt=fmt,
        title=title_base + " attacks on non-target trials",
        font_size=13,
    )

    df = []
    for i in range(len(res_dirs1)):
        file_path = "%s/%s/%s" % (base_res_dir, res_dirs1[i], filenames[2])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + "_tar_linf"
    Eval.plot_dcf_eer_vs_stat_v2(
        df,
        "n_linf",
        fig_file,
        clean_ref=clean_ref,
        xlabel=r"$L_{\infty}$",
        log_x=True,
        legends=legends,
        fmt=fmt,
        title=title_base + " attacks on target trials",
        font_size=13,
    )
    df = []
    for i in range(len(res_dirs1)):
        file_path = "%s/%s/%s" % (base_res_dir, res_dirs1[i], filenames[3])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + "_non_linf"
    Eval.plot_dcf_eer_vs_stat_v2(
        df,
        "n_linf",
        fig_file,
        clean_ref=clean_ref,
        xlabel=r"$L_{\infty}$",
        log_x=True,
        legends=legends,
        fmt=fmt,
        title=title_base + " attacks on non-target trials",
        font_size=13,
    )


def plot_figs2(
    res_dirs1,
    legends,
    title_base,
    fig_base,
    fmt=["b", "r", "g", "m", "c", "y"],
    clean_ref=0,
    colors=None,
):
    df = []
    for i in range(len(res_dirs1)):
        file_path = "%s/%s/%s" % (base_res_dir, res_dirs1[i], filenames[0])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + "_tar_snr"
    Eval.plot_dcf_eer_vs_stat_v2(
        df,
        "snr",
        fig_file,
        clean_ref=clean_ref,
        xlabel="SNR(dB)",
        higher_better=True,
        legends=legends,
        fmt=fmt,
        title=title_base + " Adv. Evasion",
        font_size=13,
        colors=colors,
    )
    df = []
    for i in range(len(res_dirs1)):
        file_path = "%s/%s/%s" % (base_res_dir, res_dirs1[i], filenames[1])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + "_non_snr"
    Eval.plot_dcf_eer_vs_stat_v2(
        df,
        "snr",
        fig_file,
        clean_ref=clean_ref,
        xlabel="SNR(dB)",
        higher_better=True,
        legends=legends,
        fmt=fmt,
        title=title_base + " Adv. Impersonation",
        font_size=13,
        colors=colors,
    )

    df = []
    for i in range(len(res_dirs1)):
        file_path = "%s/%s/%s" % (base_res_dir, res_dirs1[i], filenames[2])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + "_tar_linf"
    Eval.plot_dcf_eer_vs_stat_v2(
        df,
        "n_linf",
        fig_file,
        clean_ref=clean_ref,
        xlabel=r"$L_{\infty}$",
        log_x=True,
        legends=legends,
        fmt=fmt,
        title=title_base + " Adv. Evasion",
        font_size=13,
    )
    df = []
    for i in range(len(res_dirs1)):
        file_path = "%s/%s/%s" % (base_res_dir, res_dirs1[i], filenames[3])
        df_i = pd.read_csv(file_path, index_col=0)
        df.append(df_i)

    fig_file = output_dir + fig_base + "_non_linf"
    Eval.plot_dcf_eer_vs_stat_v2(
        df,
        "n_linf",
        fig_file,
        clean_ref=clean_ref,
        xlabel=r"$L_{\infty}$",
        log_x=True,
        legends=legends,
        fmt=fmt,
        title=title_base + " Adv. Impersonation",
        font_size=13,
    )


if __name__ == "__main__":

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    res_dirs0 = "resnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2"
    res_dirs1 = ["cosine_fgsm_eall", "cosine_randfgsm_eall", "cosine_iterfgsm_eall"]
    res_dirs1 = [res_dirs0 + "/" + s for s in res_dirs1]
    legends = ["FGSM", "Rand-FGSM", "Iter-FGSM"]
    plot_figs1(res_dirs1, legends, "FGSM", "fgsm")
    plot_figs2(res_dirs1, legends, "FGSM", "fgsm2")
    plot_figs2(res_dirs1, None, "FGSM", "fgsmnoleg2")

    res_dirs1 = [
        "cosine_cwl2_conf0",
        "cosine_cwl2_conf1",
        "cosine_cwlinf_conf0",
        "cosine_cwlinf_conf1",
    ]
    res_dirs1 = [res_dirs0 + "/" + s for s in res_dirs1]
    legends = ["CW-L2 conf=0", "CW-L2 conf=1", "CW-Linf conf=0", "CW-Linf conf=1"]
    plot_figs1(res_dirs1, legends, "Carlini-Wagner", "cw")

    ###########################

    res_dirs2 = [
        "resnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2",
        "lresnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2",
        "resetdnn_nl5ld512_e256_arcs30m0.3_do0.1_adam_lr0.05_b512_amp.v2",
    ]
    legends = ["ResNet34", "ThinResNet34", "ResETDNN"]
    res_dirs3 = [s + "/cosine_iterfgsm_eall" for s in res_dirs2]
    plot_figs1(res_dirs3, legends, "Iter-FGSM", "iterfgsm", clean_ref=None)
    plot_figs2(res_dirs3, legends, "Iter-FGSM", "iterfgsm2", clean_ref=None)
    plot_figs2(res_dirs3, None, "Iter-FGSM", "iterfgsmnoleg2", clean_ref=None)

    res_dirs3 = [s + "/cosine_cwl2_conf0" for s in res_dirs2]
    plot_figs1(res_dirs3, legends, "Carlini-Wagner L2", "cwl2", clean_ref=None)
    plot_figs2(res_dirs3, legends, "Carlini-Wagner L2", "cwl22", clean_ref=None)

    ###########################

    res_dirs1 = [
        "cosine_cwl2_conf0",
        "cosine_cwl2_conf0_noabort",
        "cosine_cwl2_conf0_lr0.001",
        "cosine_cwl2_conf0_lr0.001_noabort",
        "cosine_cwl2_conf0_lr0.001_noabort_it20",
        "cosine_cwl2_conf0_lr0.001_noabort_it40",
        "cosine_cwl2_conf0_lr0.001_noabort_it80",
        "cosine_cwl2_conf0_lr0.001_it80",
    ]
    legends = [
        "default",
        "lr=0.01 it10",
        "lr=0.001 it10 abort early",
        "lr=0.001 it10",
        "lr=0.001 it20",
        "lr=0.001 it40",
        "lr=0.001 it80",
        "lr=0.001 it80 abort early",
    ]
    fmt = ["b", "r", "g", "m", "c", "y", "*b", "*r", "*g", "*m", "*c", "*y"]

    res_dirs2 = [res_dirs0 + "/" + s for s in res_dirs1]

    plot_figs1(res_dirs2, legends, "Carlini-Wagner L2", "cwl2_iters1", fmt=fmt)

    res_dirs1 = [
        "cosine_cwl2_conf0",
        "cosine_cwl2_conf0_lr0.001_noabort",
        "cosine_cwl2_conf0_lr0.001_noabort_it20",
        "cosine_cwl2_conf0_lr0.001_noabort_it40",
        "cosine_cwl2_conf0_lr0.001_noabort_it80",
        "cosine_cwl2_conf0_lr0.001_it80",
    ]
    legends = [
        "default",
        "lr=0.001 it10",
        "lr=0.001 it20",
        "lr=0.001 it40",
        "lr=0.001 it80",
        "lr=0.001 it80 abort early",
    ]
    fmt = ["b", "r", "g", "m", "c", "y"]

    res_dirs2 = [res_dirs0 + "/" + s for s in res_dirs1]

    plot_figs1(res_dirs2, legends, "Carlini-Wagner L2", "cwl2_iters2", fmt=fmt)
    ###########################

    res_dirs0 = "resnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2"
    res_dirs1 = [
        "cosine_fgsm_eall",
        "cosine_randfgsm_eall",
        "cosine_iterfgsm_eall",
        "cosine_cwl2_conf0_lr0.001_noabort",
        "cosine_cwsnr_conf0_lr0.001_noabort_it10",
        "cosine_cwrms_conf0_lr0.001_noabort_it10",
        "cosine_cwrms_conf4_lr0.001_noabort_it10",
        "cosine_cwl2_conf0_lr0.001_noabort_it40",
    ]
    res_dirs1 = [res_dirs0 + "/" + s for s in res_dirs1]
    fmt = ["ob", "vr", "^g", ">y", "sm", "pc", "Pc", "*r", "+g", "Dc", "Hm"]
    fmt = ["ob", "vr", "^g", ">y", "sm", "pc", "P", "*", "+g", "Dc", "Hm"]
    colors = ["b", "r", "g", "y", "m", "c", "lime", "orange", "+g", "Dc", "Hm"]
    legends = [
        "FGSM",
        "Rand-FGSM",
        "Iter-FGSM",
        "CW-L2 k=0",
        "CW-SNR k=0",
        "CW-RMS k=0",
        "CW-RMS k=4",
        "CW-RMS k=0 it=40",
    ]
    legends = [
        "FGSM",
        "Rand-FGSM",
        "Iter-FGSM",
        "CW-L2",
        "CW-SNR",
        "CW-RMS",
        "CW-RMS k=4",
        "CW-RMS it=40",
    ]

    plot_figs1(res_dirs1, legends, "", "fgsmcw", fmt=fmt)
    plot_figs1(res_dirs1, None, "", "fgsmcwnoleg", fmt=fmt)
    plot_figs2(res_dirs1, legends, "", "fgsmcw2", fmt=fmt, colors=colors)
    plot_figs2(res_dirs1, None, "", "fgsmcwnoleg2", fmt=fmt, colors=colors)

    res_dirs0 = "resnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2"
    res_dirs1 = [
        "cosine_iterfgsm_eall",
        "cosine_cwl2_conf0_lr0.001_noabort",
        "cosine_cwsnr_conf0_lr0.001_noabort_it10",
        "cosine_cwrms_conf0_lr0.001_noabort_it10",
        "cosine_cwrms_conf4_lr0.001_noabort_it10",
        "cosine_cwl2_conf0_lr0.001_noabort_it40",
    ]
    res_dirs1 = [res_dirs0 + "/" + s for s in res_dirs1]
    fmt = ["ob", "vr", "^g", ">y", "sm", "pc", "Pc", "*r", "+g", "Dc", "Hm"]
    fmt = ["ob", "vr", "^g", ">y", "sm", "pc", "P", "*", "+g", "Dc", "Hm"]
    colors = ["b", "r", "g", "y", "m", "c", "lime", "orange", "+g", "Dc", "Hm"]
    legends = ["Iter-FGSM", "CW-L2", "CW-SNR", "CW-RMS", "CW-RMS k=4", "CW-RMS it=40"]

    plot_figs2(res_dirs1, legends, "", "fgsmcw3", fmt=fmt, colors=colors)
    plot_figs2(res_dirs1, None, "", "fgsmcwnoleg3", fmt=fmt, colors=colors)

    ###########################

    res_dirs1 = [
        "cosine_iterfgsm_eall",
        "cosine_iterfgsm_eall_randsmooth0.001",
        "cosine_iterfgsm_eall_randsmooth0.01",
    ]
    legends = ["no-def", "$\sigma=32$", "$\sigma=327$"]
    fmt = ["b", "r", "g", "m", "c", "y"]

    res_dirs2 = [res_dirs0 + "/" + s for s in res_dirs1]

    plot_figs1(
        res_dirs2, legends, "IterFGSM RandSmooth", "iterfgsm_randsmooth", fmt=fmt
    )
    plot_figs2(
        res_dirs2, legends, "IterFGSM RandSmooth", "iterfgsm_randsmooth2", fmt=fmt
    )
    plot_figs2(
        res_dirs2, None, "IterFGSM RandSmooth", "iterfgsm_randsmoothnoleg2", fmt=fmt
    )

    ###########################

    res_dirs2 = [
        "resnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2",
        "resnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2.advft_400_400_sgdcos_lr0.005_b256_attack_p0.5eps1step0.125_amp.v1_ep5",
        "resnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2.advft_400_400_sgdcos_lr0.005_b256_attack_p0.5eps1step0.125_amp.v1",
    ]
    legends = ["No-adv", "Adv. epoch=5", "Adv. epoch=23"]
    res_dirs3 = [s + "/cosine_fgsm_eall" for s in res_dirs2]
    plot_figs1(res_dirs3, legends, "FGSM adv. finetuning", "fgsm_advft", clean_ref=None)

    ###########################

    res_dirs1 = [
        "cosine_fgsm_eall",
        "cosine_randfgsm_eall",
        "cosine_iterfgsm_eall",
        "cosine_cwl2_conf0",
        "cosine_cwl2_conf1",
        "cosine_cwlinf_conf0",
        "cosine_cwlinf_conf1",
    ]
    names = [
        "FGSM",
        "Rand-FGSM",
        "Iter-FGSM",
        "CW-L2 conf=0",
        "CW-L2 conf=1",
        "CW-Linf conf=0",
        "CW-Linf conf=1",
    ]
    fig_names = [
        "fgsm",
        "randfgsm",
        "iterfgsm",
        "cwl2_conf0",
        "cwl2_conf1",
        "cwlinf_conf0",
        "cwlinf_conf1",
    ]
    legends = ["ResNet34 (white-box)", "ThinResNet34", "ResETDNN"]
    fmt = ["b", "r", "g", "m", "c", "y"]
    for i in range(len(names)):
        res_dirs2 = [
            res_dirs1[i],
            "transfer.lresnet34_zir_e256_arc0.3_do0_adam_lr0.05_b512.v2/"
            + res_dirs1[i],
            "transfer.resetdnn_nl5ld512_e256_arcs30m0.3_do0.1_adam_lr0.05_b512_amp.v2/"
            + res_dirs1[i],
        ]
        res_dirs2 = [res_dirs0 + "/" + s for s in res_dirs2]
        plot_figs1(
            res_dirs2, legends, names[i] + " black-box", fig_names[i] + "_bbox", fmt=fmt
        )
        plot_figs2(
            res_dirs2,
            legends,
            names[i] + " black-box",
            fig_names[i] + "_bbox2",
            fmt=fmt,
        )
