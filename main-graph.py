import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Número de muestras
N = 64
t = np.arange(N)

# Señal original
x = np.sin(2 * np.pi * 5 * t / N) + 0.5 * np.sin(2 * np.pi * 15 * t / N)

# Calcular la DFT
X = np.fft.fft(x)
frequencies = np.fft.fftfreq(N)

# Configuración de la figura
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Configuración del círculo (representación de coeficientes de Fourier)
ax[0].set_xlim(-1.5, 1.5)
ax[0].set_ylim(-1.5, 1.5)
ax[0].set_title('Movimiento del Punto en el Círculo')
ax[0].set_xticks([])
ax[0].set_yticks([])
circle = plt.Circle((0, 0), 1, fill=False)
ax[0].add_patch(circle)
point, = ax[0].plot([], [], 'ro', markersize=8)

# Configuración de la señal reconstruida
ax[1].set_xlim(0, N)
ax[1].set_ylim(-2, 2)  # Se ajustará dinámicamente en la animación
ax[1].set_title('Reconstrucción de la señal')
ax[1].set_xlabel('Tiempo')
ax[1].set_ylabel('Amplitud')
reconstructed_x = np.arange(N)
reconstructed_y = np.zeros(N)
wave_line, = ax[1].plot(reconstructed_x, reconstructed_y, 'b')

# Texto para mostrar el valor de k y la ecuación
text_k = ax[0].text(-1, 1.3, '', fontsize=12, color='blue')
text_equation = ax[1].text(0, 1.5, '', fontsize=12, color='red')

# Variables de animación
frames_per_k = 30  # Cambia cada 3 segundos
k_values = np.arange(min(11, N // 2))  # Limitar k a un máximo de 10
k_index = 0


# Función de actualización
def update(frame):
    global k_index, reconstructed_y

    if frame % frames_per_k == 0:  # Cambiar k cada 3 segundos
        k_index = (k_index + 1) % len(k_values)

    k = k_values[k_index]
    coef = X[k] / N  # Coeficiente normalizado

    # Movimiento continuo del punto en el círculo
    # Movimiento continuo del punto en el círculo
    x_point = np.cos(2 * np.pi * k * frame / N)
    y_point = np.sin(2 * np.pi * k * frame / N)

    point.set_data([x_point], [y_point])  # Actualiza la posición del punto

    # Actualizar la reconstrucción de la señal
    reconstructed_y = np.zeros(N)  # Reiniciar antes de acumular términos
    for i in range(k_index + 1):  # Sumar coeficientes hasta k actual
        coef_i = X[i] / N
        angle_i = 2 * np.pi * i * t / N
        reconstructed_y += 2 * np.real(coef_i) * np.cos(angle_i)

    wave_line.set_ydata(reconstructed_y)

    # Ajustar automáticamente los límites del eje Y
    ax[1].set_ylim(-max(abs(reconstructed_y)) * 1.2, max(abs(reconstructed_y)) * 1.2)

    # Mostrar el valor de k y la ecuación de la onda
    text_k.set_text(f'k = {k}')
    text_equation.set_text(f'Función aproximada con {k} términos')

    return point, wave_line, text_k, text_equation


# Animación
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
