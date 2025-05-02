# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# https://docs.python.org/3/library/timeit.html

import numpy as np
import timeit

# calculate the discrete Fourier transform directly.
def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)
# fast Fourier transform works by computing the discrete Fourier transform for small subsets of the overall problem and then combining the results.
def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    if N <= 2:
        return dft(x) if N == 2 else x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])
# Fourier transform. Only this time around, we make use of vector operations instead of recursion.
def fft_v(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")

    N_min = min(N, 2)

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                       / X.shape[0])[:, None]
        X = np.vstack([X_even + terms * X_odd,
                       X_even - terms * X_odd])
    return X.ravel()

if __name__ == '__main__':
    x = np.random.random(1024)
    #NumPy implementation uses matrix operations to calculate the Fourier transforms
    print(np.allclose(dft(x), np.fft.fft(x)))
    print(np.allclose(fft(x), np.fft.fft(x)))
    print(np.allclose(fft_v(x), np.fft.fft(x)))

    # Measure execution time using timeit
    dft_time = timeit.timeit('dft(x)', setup='from __main__ import dft, x', number=10)
    fft_time = timeit.timeit('fft(x)', setup='from __main__ import fft, x', number=10)
    fft_v_time = timeit.timeit('fft_v(x)', setup='from __main__ import fft_v, x', number=10)
    numpy_fft_time = timeit.timeit('np.fft.fft(x)', setup='import numpy as np; from __main__ import x', number=10)

    print(f"DFT execution time: {dft_time} seconds")
    print(f"FFT execution time: {fft_time} seconds")
    print(f"FFT (vectorized) execution time: {fft_v_time} seconds")
    print(f"FFT (numpy) execution time: {numpy_fft_time} seconds")

