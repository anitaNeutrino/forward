import os.path as op
from typing import Tuple

import numpy as np
from oct2py import octave

import forward

# set the directory where we store the MATLAB files
octavedir = op.join(op.abspath(op.dirname(op.dirname(__file__))), "octave")

# and add it to the MATLAB PATH
octave.addpath(octavedir)


def wienforwd(
    signal: np.ndarray,
    response: np.ndarray,
    wtype: forward.WaveletType,
    p: int,
    sigma: np.ndarray,
    scaling: np.ndarray,
    rho: np.ndarray,
    rule: forward.ThresholdRule,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Wiener deconvolution.

    Retuns the wavelet coefficients, the ratio of
    unthresholded coefficents, and the threshold vector.
    """

    return octave.wienforwd(
        signal,
        response,
        forward.wtype_as_string(wtype),
        p,
        sigma,
        scaling,
        rho,
        forward.rule_as_string(rule),
        nout=3,
    )


def waveletDecon(
    signal: np.ndarray,
    response: np.ndarray,
    sigma: np.ndarray,
    scaling: np.ndarray,
    rho: np.ndarray,
    p: int = 5,
    wtype: forward.WaveletType = forward.WaveletType.Meyer,
    rule: forward.ThresholdRule = forward.ThresholdRule.Soft,
) -> np.ndarray:
    """
    Perform wavelet deconvolution using FoRWarD.
    """

    # perform the deconvolution into the wavelet basis
    w, ratiothresh, thresh = wienforwd(
        signal, response, wtype, p=p, sigma=sigma, scaling=scaling, rho=rho, rule=rule
    )

    # and perform the inverse transformation
    return octave.iwtrans(w, forward.wtype_as_string(wtype), p)[0, :]
