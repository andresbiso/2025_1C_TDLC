import numpy as np
import cv2
from scipy.fft import fft2 as scipy_fft2, ifft2 as scipy_ifft2
from numpy.fft import fft2 as numpy_fft2, ifft2 as numpy_ifft2
import tensorflow as tf
import torch
import timeit
import json
import matplotlib.pyplot as plt

# Genera una señal 2D con ruido
def generar_senal(tamano):
    np.random.seed(0)
    x = np.linspace(0, 1, tamano)
    y = np.linspace(0, 1, tamano)
    X, Y = np.meshgrid(x, y)
    senal = np.sin(2 * np.pi * 50 * X) + np.sin(2 * np.pi * 120 * Y) + 0.5 * np.random.randn(tamano, tamano)
    return senal.astype(np.float32)

# Compara las implementaciones FFT en tiempo y precisión para un tamaño dado
def comparar_fft(tamano):
    senal = generar_senal(tamano)

    # Funciones FFT/IFFT específicas de cada biblioteca
    scipy_fft_func, scipy_ifft_func = scipy_fft2, scipy_ifft2
    numpy_fft_func, numpy_ifft_func = numpy_fft2, numpy_ifft2
    tf_fft_func, tf_ifft_func = tf.signal.fft2d, tf.signal.ifft2d
    cv2_fft_func = lambda x: cv2.dft(np.float32(x), flags=cv2.DFT_COMPLEX_OUTPUT)
    cv2_ifft_func = lambda x: cv2.idft(x, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    torch_fft_func = lambda x: torch.fft.fft2(x)
    torch_ifft_func = lambda x: torch.fft.ifft2(x)

    # SciPy
    scipy_fft_time = timeit.timeit(lambda: scipy_fft_func(senal), number=10)
    scipy_fft = scipy_fft_func(senal)
    scipy_ifft_time = timeit.timeit(lambda: scipy_ifft_func(scipy_fft), number=10)
    scipy_reconstruida = scipy_ifft_func(scipy_fft)
    scipy_mse = np.mean((senal - scipy_reconstruida.real) ** 2)

    # NumPy
    numpy_fft_time = timeit.timeit(lambda: numpy_fft_func(senal), number=10)
    numpy_fft = numpy_fft_func(senal)
    numpy_ifft_time = timeit.timeit(lambda: numpy_ifft_func(numpy_fft), number=10)
    numpy_reconstruida = numpy_ifft_func(numpy_fft)
    numpy_mse = np.mean((senal - numpy_reconstruida.real) ** 2)

    # TensorFlow
    senal_tf = tf.convert_to_tensor(senal)
    tf_fft_time = timeit.timeit(lambda: tf_fft_func(tf.cast(senal_tf, tf.complex64)), number=10)
    tf_fft = tf_fft_func(tf.cast(senal_tf, tf.complex64))
    tf_ifft_time = timeit.timeit(lambda: tf_ifft_func(tf_fft), number=10)
    tf_reconstruida = tf_ifft_func(tf_fft_func(tf.cast(senal_tf, tf.complex64)))
    tf_mse = np.mean((senal - tf_reconstruida.numpy().real) ** 2)

    # PyTorch
    senal_torch = torch.tensor(senal).to("cuda" if torch.cuda.is_available() else "cpu")
    torch_fft_time = timeit.timeit(lambda: torch_fft_func(senal_torch), number=10)
    torch_fft = torch_fft_func(senal_torch)
    torch_ifft_time = timeit.timeit(lambda: torch_ifft_func(torch_fft), number=10)
    torch_reconstruida = torch_ifft_func(torch_fft_func(senal_torch)).to("cpu")
    torch_mse = np.mean((senal - torch_reconstruida.numpy().real) ** 2)

    # OpenCV
    cv2_fft_time = timeit.timeit(lambda: cv2_fft_func(senal), number=10)
    cv2_fft = cv2_fft_func(senal)
    cv2_ifft_time = timeit.timeit(lambda: cv2_ifft_func(cv2_fft), number=10)
    cv2_reconstruida = cv2_ifft_func(cv2_fft_func(senal))
    cv2_mse = np.mean((senal - cv2_reconstruida) ** 2)

    # Resultados
    return {
        'scipy_fft_time': scipy_fft_time, 'scipy_ifft_time': scipy_ifft_time, 'scipy_mse': scipy_mse,
        'numpy_fft_time': numpy_fft_time, 'numpy_ifft_time': numpy_ifft_time, 'numpy_mse': numpy_mse,
        'tf_fft_time': tf_fft_time, 'tf_ifft_time': tf_ifft_time, 'tf_mse': tf_mse,
        'cv2_fft_time': cv2_fft_time, 'cv2_ifft_time': cv2_ifft_time, 'cv2_mse': cv2_mse,
        'torch_fft_time': torch_fft_time, 'torch_ifft_time': torch_ifft_time, 'torch_mse': torch_mse
    }

# Bloque principal
if __name__ == "__main__":
    tamanos = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    resultados = {
        'scipy_fft_time': [], 'scipy_ifft_time': [], 'scipy_mse': [],
        'numpy_fft_time': [], 'numpy_ifft_time': [], 'numpy_mse': [],
        'tf_fft_time': [], 'tf_ifft_time': [], 'tf_mse': [],
        'cv2_fft_time': [], 'cv2_ifft_time': [], 'cv2_mse': [],
        'torch_fft_time': [], 'torch_ifft_time': [], 'torch_mse': []
    }

    for tamano in tamanos:
        print(f"Procesando tamaño: {tamano}")
        resultado = comparar_fft(tamano)
        for clave in resultado:
            resultados[clave].append(resultado[clave])
        print(f"Completado tamaño: {tamano}")

    with open('fft_results.json', 'w') as archivo:
        copia_resultados = {k: [float(x) for x in v] for k, v in resultados.items()}
        copia_resultados['sizes'] = tamanos
        json.dump(copia_resultados, archivo, indent=4)
        print("Resultados guardados en fft_results.json")

    with open('fft_results.json', 'r') as archivo:
        resultados = json.load(archivo)

    tiempos_fft = {
        "SciPy": resultados["scipy_fft_time"],
        "NumPy": resultados["numpy_fft_time"],
        "TensorFlow": resultados["tf_fft_time"],
        "OpenCV": resultados["cv2_fft_time"],
        "PyTorch": resultados["torch_fft_time"],
    }

    plt.figure(figsize=(10, 6))
    for libreria, tiempos in tiempos_fft.items():
        plt.plot(tamanos, tiempos, marker='o', label=libreria)

    plt.xlabel("Tamaño de la Señal")
    plt.ylabel("FFT - Tiempo de Ejecución (en segundos)")
    plt.title("FFT - Tiempo de Ejecución vs Tamaño de la Señal")
    plt.legend()
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("fft_execution_times.png", dpi=300, bbox_inches="tight")
    print("Gráfico guardado como fft_execution_times.png")

    # Gráfico: Tiempo de ejecución IFFT
    tiempos_ifft = {
        "SciPy": resultados["scipy_ifft_time"],
        "NumPy": resultados["numpy_ifft_time"],
        "TensorFlow": resultados["tf_ifft_time"],
        "OpenCV": resultados["cv2_ifft_time"],
        "PyTorch": resultados["torch_ifft_time"],
    }

    plt.figure(figsize=(10, 6))
    for libreria, tiempos in tiempos_ifft.items():
        plt.plot(tamanos, tiempos, marker='s', label=libreria)

    plt.xlabel("Tamaño de la Señal")
    plt.ylabel("IFFT - Tiempo de Ejecución (en segundos)")
    plt.title("IFFT - Tiempo de Ejecución vs Tamaño de la Señal")
    plt.legend()
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("ifft_execution_times.png", dpi=300, bbox_inches="tight")
    print("Gráfico guardado como ifft_execution_times.png")

    # Gráfico: Error MSE
    errores_mse = {
        "SciPy": resultados["scipy_mse"],
        "NumPy": resultados["numpy_mse"],
        "TensorFlow": resultados["tf_mse"],
        "OpenCV": resultados["cv2_mse"],
        "PyTorch": resultados["torch_mse"],
    }

    plt.figure(figsize=(10, 6))
    for libreria, errores in errores_mse.items():
        plt.plot(tamanos, errores, marker='^', label=libreria)

    plt.xlabel("Tamaño de la Señal")
    plt.ylabel("Error Cuadrático Medio (MSE)")
    plt.title("Precisión de Reconstrucción vs Tamaño de la Señal")
    plt.legend()
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("mse_comparison.png", dpi=300, bbox_inches="tight")
    print("Gráfico guardado como mse_comparison.png")
