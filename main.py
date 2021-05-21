# %%
'''
# 1. Importando librerias necesarias.
'''
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np


# %%
'''
# 2. Guardando el nombre de la pista de audio en la variable "audio_data".
'''
audio_data = 'sample.wav'


# %%
'''
# 3. Con la función load() que recibe el nombre de la pista de audio ("audio_data") obtenemos:
    - x: En la variable x se carga y decodifica el audio como una matriz de punto flotante NumPy unidimensional.
    - sr: La variable sr contiene la frecuencia de muestreo de x, es decir, el número de muestras por segundo del audio.
'''
x, sr = librosa.load(audio_data)


# %%
'''
# 4. Reproduciendo automaticamente el audio con la ayuda de la clase Audio().
'''
Audio(audio_data, autoplay=True)


# %%
'''
# 5. Trazar la señal muestreada.
    - Con la función figure() de matplotlib creamos una nueva figura.
    Con el parámetro "figsize" indicamos el ancho y alto en pulgadas de la figura.
    - Luego de creada la figura, se traza la envolvente de amplitud de una forma de onda mediante la función waveplot().
    waveplot() recibe como parametros:
        • x: variable de decodificación de audio.
        • sr: frecuencia de muestreo de la variable "x".
'''
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# %%
'''
# 6. Modificando la velocidad a x2 (Rapida)
    - speed: variable que define la velocidad a la que se modificará la señal de audio.
    - x_speed_changed: obtiene la matriz del audio modificado en su velocidad.
    Esto gracias a la función time_stretch(), la cual recive la matriz del audio y como segundo parámetro la velocidad.
'''

speed = 2.0  # (x2)
x_speed_changed = librosa.effects.time_stretch(
    x.astype('float64'), speed)

# Reproduciendo el audio modificado en la velocidad
Audio(x_speed_changed, rate=sr)

# %%
'''
# 7. Trazar la señal muestreada del audio modificado.
'''
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x_speed_changed, sr=sr)
