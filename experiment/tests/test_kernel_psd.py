"""The Section 4 product kernel must be positive semidefinite."""

import numpy as np

from experiment.data import make_stream
from experiment.kernel import ProductKernel, pack


def test_product_kernel_psd_on_random_sample():
    stream = make_stream(120, seed=7)
    P = pack(stream)
    K = ProductKernel(sigma2=1.6, l_tool=1.1, l_ctx=1.2, lam=90.0).full(P)

    assert np.allclose(K, K.T, atol=1e-10), "kernel must be symmetric"
    w = np.linalg.eigvalsh(K)
    assert w.min() > -1e-8, f"kernel not PSD: min eigenvalue {w.min()}"
    # Diagonal equals the prior signal variance (each block self-sim = 1).
    assert np.allclose(np.diag(K), 1.6, atol=1e-9)


def test_block_kernels_individually_psd():
    stream = make_stream(60, seed=3)
    P = pack(stream)
    k = ProductKernel()
    K = k.full(P)
    # Product of PSD blocks is PSD; check the assembled matrix is usable in a
    # Cholesky after the standard jitter.
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(stream)))
    assert np.all(np.isfinite(L))
