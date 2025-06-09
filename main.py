# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# https://docs.python.org/3/library/timeit.html
'''
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
'''

import numpy as np
import cv2
from scipy.fft import fft2 as scipy_fft2, ifft2 as scipy_ifft2
from numpy.fft import fft2 as numpy_fft2, ifft2 as numpy_ifft2
import tensorflow as tf
import torch
import timeit
import json


# Function to generate 2D signals with noise
def generate_signal(size):
    np.random.seed(0)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    signal = np.sin(2 * np.pi * 50 * X) + np.sin(2 * np.pi * 120 * Y) + 0.5 * np.random.randn(size, size)
    return signal.astype(np.float32)


# Function to run the FFT and IFFT for all libraries
def compare_fft(size):
    signal = generate_signal(size)

    # Define FFT and IFFT functions
    scipy_fft_func, scipy_ifft_func = scipy_fft2, scipy_ifft2
    numpy_fft_func, numpy_ifft_func = numpy_fft2, numpy_ifft2
    tf_fft_func, tf_ifft_func = tf.signal.fft2d, tf.signal.ifft2d
    cv2_fft_func = lambda x: cv2.dft(np.float32(x), flags=cv2.DFT_COMPLEX_OUTPUT)
    cv2_ifft_func = lambda x: cv2.idft(x, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    torch_fft_func = lambda x: torch.fft.fft2(x)
    torch_ifft_func = lambda x: torch.fft.ifft2(x)

    # Time and accuracy for scipy FFT and IFFT
    scipy_fft_time = timeit.timeit(lambda: scipy_fft_func(signal), number=10)
    scipy_fft = scipy_fft_func(signal)
    scipy_ifft_time = timeit.timeit(lambda: scipy_ifft_func(scipy_fft), number=10)
    scipy_transformed_back = scipy_ifft_func(scipy_fft)
    scipy_mse = np.mean((signal - scipy_transformed_back.real) ** 2)

    # Time and accuracy for numpy FFT and IFFT
    numpy_fft_time = timeit.timeit(lambda: numpy_fft_func(signal), number=10)
    numpy_fft = numpy_fft_func(signal)
    numpy_ifft_time = timeit.timeit(lambda: numpy_ifft_func(numpy_fft), number=10)
    numpy_transformed_back = numpy_ifft_func(numpy_fft)
    numpy_mse = np.mean((signal - numpy_transformed_back.real) ** 2)

    # Convert numpy array to TensorFlow tensor
    signal_tf = tf.convert_to_tensor(signal)

    # Time and accuracy for tensorflow FFT and IFFT
    tf_fft_time = timeit.timeit(lambda: tf_fft_func(tf.cast(signal_tf, tf.complex64)), number=10)
    tf_fft = tf_fft_func(tf.cast(signal_tf, tf.complex64))
    tf_ifft_time = timeit.timeit(lambda: tf_ifft_func(tf_fft), number=10)
    tf_transformed_back = tf_ifft_func(tf_fft_func(tf.cast(signal_tf, tf.complex64)))
    tf_mse = np.mean((signal - tf_transformed_back.numpy().real) ** 2)

    # Time and accuracy for torch FFT and IFFT
    signal_torch = torch.tensor(signal).to("cuda" if torch.cuda.is_available() else "cpu")
    torch_fft_time = timeit.timeit(lambda: torch_fft_func(signal_torch), number=10)
    torch_fft = torch_fft_func(signal_torch)
    torch_ifft_time = timeit.timeit(lambda: torch_ifft_func(torch_fft), number=10)
    torch_transformed_back = torch_ifft_func(torch_fft_func(signal_torch)).to("cpu")
    torch_mse = np.mean((signal - torch_transformed_back.numpy().real) ** 2)

    # Time and accuracy for cv2 FFT and IFFT
    cv2_fft_time = timeit.timeit(lambda: cv2_fft_func(signal), number=10)
    cv2_fft = cv2_fft_func(signal)
    cv2_ifft_time = timeit.timeit(lambda: cv2_ifft_func(cv2_fft), number=10)
    cv2_transformed_back = cv2_ifft_func(cv2_fft_func(signal))
    cv2_mse = np.mean((signal - cv2_transformed_back) ** 2)

    return {
        'scipy_fft_time': scipy_fft_time, 'scipy_ifft_time': scipy_ifft_time, 'scipy_mse': scipy_mse,
        'numpy_fft_time': numpy_fft_time, 'numpy_ifft_time': numpy_ifft_time, 'numpy_mse': numpy_mse,
        'tf_fft_time': tf_fft_time, 'tf_ifft_time': tf_ifft_time, 'tf_mse': tf_mse,
        'cv2_fft_time': cv2_fft_time, 'cv2_ifft_time': cv2_ifft_time, 'cv2_mse': cv2_mse,
        'torch_fft_time': torch_fft_time, 'torch_ifft_time': torch_ifft_time, 'torch_mse': torch_mse
    }


# Test with different sizes for 2D signals
sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
results = {
    'scipy_fft_time': [], 'scipy_ifft_time': [], 'scipy_mse': [],
    'numpy_fft_time': [], 'numpy_ifft_time': [], 'numpy_mse': [],
    'tf_fft_time': [], 'tf_ifft_time': [], 'tf_mse': [],
    'cv2_fft_time': [], 'cv2_ifft_time': [], 'cv2_mse': [],
    'torch_fft_time': [], 'torch_ifft_time': [], 'torch_mse': []
}

for size in sizes:
    print(f"Processing size: {size}")
    result = compare_fft(size)
    for key in result:
        results[key].append(result[key])
    print(f"Completed size: {size}")

# Save the results to a JSON file
with open('fft_results.json', 'w') as f:
    results_copy = {k: [float(x) for x in v] for k, v in results.items()}
    results_copy['sizes'] = sizes
    json.dump(results_copy, f, indent=4)
    print("Results saved to fft_results.json")
