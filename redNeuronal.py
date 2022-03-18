import tensorflow.python.keras.optimizers as optimizador
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import tensorflow as tf

# Eliminamos otros entrenamientos
K.clear_session()

# Obtenemos los datos de validacion y entrenamiento
datos_entrenamiento = "C:/Users/David/Documents/Semestre_2021_2/Python/PF/Entrenamiento"
datos_validacion = "C:/Users/David/Documents/Semestre_2021_2/Python/PF/Validacion"

# Parametros de CNN
iteraciones = 30  # No. de veces que realizara el entrenamiento y validacion
altura, longitud = 100, 100  # Dimenciones de las fotos (datos de entrada)
batch_size = 1  # Cuantos datos tomara
pasos = 300/1  # No. de veces que entrenara
pasos_validacion = 300/1  # No. de veces que validara

# Cantidad de neuronas por filtro
filtrosConv1 = 32
filtrosConv2 = 64
filtrosConv3 = 128
filtrosConv4 = 256

# Tamaño de los filstros
tam_filtro1 = (5, 5)
tam_filtro2 = (4, 4)
tam_filtro3 = (3, 3)
tam_filtro4 = (2, 2)
tam_pool = (2, 2)

# No. de clases a clasificar
clases = 5
# Rango de aprendizaje
lr = 0.0005

# Modificacion de las fotos para un mejor entrenamiento
preprocesamiento_entre = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

# Modificacion de las fotos para una mejor validacion
preprocesamiento_val = ImageDataGenerator(
    rescale=1./255
)

# Pasamos los parametros de las fotos de entrenamiento
imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)
# Pasamos los parametros de las fotos de validacion
imagen_validacion = preprocesamiento_val.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

cnn = Sequential()  # Indicamos que tipo de red neuronal usaremos

# Capas de la red neuronal
cnn.add(Convolution2D(filtrosConv1,
        tam_filtro1,
        padding='same',
        input_shape=(altura, longitud, 3),
        activation='relu'))

cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Convolution2D(filtrosConv2,
        tam_filtro2,
        padding='same',
        activation='relu'))

cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Convolution2D(filtrosConv3, tam_filtro3,
        padding='same', activation='relu'))

cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Convolution2D(filtrosConv4, tam_filtro4,
        padding='same', activation='relu'))

cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Flatten())  # Aplanamos las capas

# Indicamos que tan densa sera la red y la funcion de activacion
cnn.add(Dense(980, activation='relu'))
cnn.add(Dropout(0.5))  # Indicamos que apage neuronas aletoreamente
# Indicamos las clases y la funcion de activacion
cnn.add(Dense(clases, activation='softmax'))

optimizar = optimizador.adam_v2.Adam(
    learning_rate=lr)  # Optimizador de la red neuronal

cnn.compile(optimizer=optimizar,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

cnn.fit(imagen_entreno, steps_per_epoch=pasos, epochs=iteraciones,
        validation_data=imagen_validacion, validation_steps=pasos_validacion)

# Guardamos el modelo y pesos de la red para su uso
cnn.save('ModeloVocales.h5')
cnn.save_weights('PesosVocales.h5')
