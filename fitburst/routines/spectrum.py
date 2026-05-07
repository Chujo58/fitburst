"""
Routines for Spectral Energy Distributions (SEDs)

This module contains functions that return SED values that are
derivable from analytic expression. These funtions are used to
derive the frequency variation of the dynamic spectrum.
"""

import numpy as np


def compute_spectrum_rpl(
    freqs: np.ndarray, freq_ref: float, sp_idx: float, sp_run: float
) -> float:
    """
    Computes a one-dimensional frequency spectrum assuming the form of a
    running power law (rpl).

    Parameters
    ----------
    freqs : np.ndarray
        an array of observing frequencies at which to evaluate spectrum

    freq_ref : float
        a reference frequency used for normalization

    sp_idx : float
        spectral index of spectrum

    sp_run : float
        the 'running' of the spectral index, characterizing first-order
        devations from the basic power-law form.

    Returns
    -------
    spectrum : np.ndarray
        the one-dimensional spectrum for input frequencies

    """

    sp_idx = np.array(sp_idx)
    sp_run = np.array(sp_run)

    log_freq = np.tile(
        np.log(freqs / freq_ref).reshape(len(freqs), 1), (1, len(sp_idx))
    )
    exponent = log_freq * sp_idx[None, :] + log_freq**2 * sp_run[None, :]
    spectrum = np.exp(exponent)
    return spectrum
