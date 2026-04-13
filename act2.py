
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Definición del tiempo
# -------------------------------
t = np.linspace(-1, 1, 1000)
dt = t[1] - t[0]

# -------------------------------
# 2. Señales en el dominio del tiempo
# -------------------------------

# Señal senoidal
f = 5  # frecuencia en Hz
senal_seno = np.sin(2 * np.pi * f * t)

# Pulso rectangular
senal_rect = np.where(np.abs(t) < 0.2, 1, 0)

# Función escalón
senal_step = np.where(t >= 0, 1, 0)

# Señal amortiguada (extra)
senal_amort = np.exp(-3 * np.abs(t)) * np.sin(2 * np.pi * f * t)

# -------------------------------
# 3. Función para calcular FFT
# -------------------------------
def calcular_fft(senal):
    N = len(senal)
    fft = np.fft.fft(senal)
    fft = np.fft.fftshift(fft)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    return freq, fft

# -------------------------------
# 4. Función para graficar
# -------------------------------
def graficar(t, senal, freq, fft, titulo):
    plt.figure(figsize=(12,5))

    # Dominio del tiempo
    plt.subplot(1,2,1)
    plt.plot(t, senal)
    plt.title(f"{titulo} (Tiempo)")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")

    # Dominio de la frecuencia
    plt.subplot(1,2,2)
    plt.plot(freq, np.abs(fft))
    plt.title(f"{titulo} (Frecuencia)")
    plt.xlabel("Frecuencia")
    plt.ylabel("Magnitud")

    plt.tight_layout()
    plt.show()

# -------------------------------
# 5. FFT de señales básicas
# -------------------------------
freq_seno, fft_seno = calcular_fft(senal_seno)
freq_rect, fft_rect = calcular_fft(senal_rect)
freq_step, fft_step = calcular_fft(senal_step)
freq_amort, fft_amort = calcular_fft(senal_amort)

# -------------------------------
# 6. Graficar señales básicas
# -------------------------------
graficar(t, senal_seno, freq_seno, fft_seno, "Señal Senoidal")
graficar(t, senal_rect, freq_rect, fft_rect, "Pulso Rectangular")
graficar(t, senal_step, freq_step, fft_step, "Función Escalón")
graficar(t, senal_amort, freq_amort, fft_amort, "Señal Amortiguada")

# -------------------------------
# 7. Propiedades de Fourier
# -------------------------------
# Verificación de propiedades de la Transformada de Fourier

# ✔️ Linealidad
senal_suma = senal_seno + senal_rect
freq_suma, fft_suma = calcular_fft(senal_suma)
graficar(t, senal_suma, freq_suma, fft_suma, "Linealidad (Suma de señales)")

# ✔️ Desplazamiento en el tiempo
senal_shift = np.sin(2 * np.pi * f * (t - 0.2))
freq_shift, fft_shift = calcular_fft(senal_shift)
graficar(t, senal_shift, freq_shift, fft_shift, "Desplazamiento en el tiempo")

# ✔️ Escalamiento en el tiempo
senal_scale = np.sin(2 * np.pi * f * (2 * t))
freq_scale, fft_scale = calcular_fft(senal_scale)
graficar(t, senal_scale, freq_scale, fft_scale, "Escalamiento en el tiempo")
