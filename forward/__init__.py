from _forward import *  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt


def plot_coeffs(w: np.ndarray, p: int) -> None:
    """
    Plot the multi-stage wavelet coefficients.
    """

    # create the subplots
    fig, axes = plt.subplots(ncols=1, nrows=p + 1)

    # loop through the different wavelet stages
    for q in 1 + np.arange(p + 1):

        # get the coefficients
        coeffs = coeff(w, p, q)

        # make sure it is real
        if coeffs.dtype == np.complex:
            coeffs = coeffs.real

        # plot the coefficients
        markers, stemlines, baseline = axes[q - 1].stem(
            coeffs.real, label=f"Stage: {p+1}", use_line_collection=True
        )

        # configre the stem plot
        plt.setp(stemlines, linewidth=0.5)
        plt.setp(markers, markersize=0.5)
        plt.setp(baseline, linewidth=0.2)

        # and make some more space available
        axes[q - 1].xaxis.set_ticklabels([])


def rule_as_string(rule: ThresholdRule) -> str:
    """
    """
    if rule == ThresholdRule.Soft:
        return "soft"
    elif rule == ThresholdRule.Hard:
        return "hard"
    else:
        raise ValueError(f"Unknown threshold rule.")


def wtype_as_string(wtype: WaveletType) -> str:
    """
    """
    if wtype == WaveletType.d10:
        return "d10"
    elif wtype == WaveletType.d12:
        return "d12"
    elif wtype == WaveletType.d14:
        return "d14"
    elif wtype == WaveletType.d16:
        return "d16"
    elif wtype == WaveletType.d18:
        return "d18"
    elif wtype == WaveletType.d20:
        return "d20"
    elif wtype == WaveletType.Meyer:
        return "meyer"
    else:
        raise ValueError(f"Unknown wavelet type.")
