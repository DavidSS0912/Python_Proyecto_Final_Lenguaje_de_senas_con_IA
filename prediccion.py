import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

modelo = "C:/Users/David/Documents/Semestre_2021_2/Python/PF/ModeloVocales.h5"
pesos = "C:/Users/David/Documents/Semestre_2021_2/Python/PF/PesosVocales.h5"

# Cargamos los pesos y modelos
cnn = load_model(modelo)
cnn.load_weights(pesos)


cap = cv2.VideoCapture(0)

# Reconocimeinto de manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()
dibujo = mp.solutions.drawing_utils

while(1):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corX, corY = int(lm.x*ancho), int(lm.y * alto)
                posiciones.append([id, corX, corY])
                dibujo.draw_landmarks(
                    frame, mano, clase_manos.HAND_CONNECTIONS)

            if len(posiciones) != 0:
                p1 = posiciones[3]
                p2 = posiciones[17]
                p3 = posiciones[10]
                p4 = posiciones[0]
                p5 = posiciones[9]

                x1, y1 = (p5[1] - 100), (p5[2] - 100)
                ancho, alto = (x1 + 100), (y1 + 100)
                x2, y2 = x1 + ancho, y1 + alto

                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1-50, y1-50),
                              (x2-50, y2+50), (0, 0, 255), 3)
                dedos_reg = cv2.resize(dedos_reg, (100, 100),
                                       interpolation=cv2.INTER_CUBIC)

                x = img_to_array(dedos_reg)
                x = np.expand_dims(x, axis=0)

                vector = cnn.predict(x)
                resultado = vector[0]
                respuesta = np.argmax(resultado)

                if respuesta == 0:
                    print(resultado, " = A")
                    cv2.rectangle(frame, (x1-50, y1-50),
                                  (x2-50, y2+50), (0, 255, 0), 3)
                    cv2.putText(frame,  "A", (x1-10, y1 - 5), 1,
                                2, (0, 255, 0), 3, cv2.LINE_AA)
                elif respuesta == 1:
                    print(resultado, " = E")
                    cv2.rectangle(frame, (x1-50, y1-50),
                                  (x2-50, y2+50), (0, 255, 0), 3)
                    cv2.putText(frame,  "E", (x1-10, y1 - 5), 1,
                                2, (0, 255, 0), 3, cv2.LINE_AA)
                elif respuesta == 2:
                    print(resultado, " = I")
                    cv2.rectangle(frame, (x1-50, y1-50),
                                  (x2-50, y2+50), (0, 255, 0), 3)
                    cv2.putText(frame,  "I", (x1-10, y1 - 5), 1,
                                2, (0, 255, 0), 3, cv2.LINE_AA)
                elif respuesta == 3:
                    print(resultado, " = O")
                    cv2.rectangle(frame, (x1-50, y1-50),
                                  (x2-50, y2+50), (0, 255, 0), 3)
                    cv2.putText(frame,  "O", (x1-10, y1 - 5), 1,
                                2, (0, 255, 0), 3, cv2.LINE_AA)
                elif respuesta == 4:
                    print(resultado, " = U")
                    cv2.rectangle(frame, (x1-50, y1-50),
                                  (x2-50, y2+50), (0, 255, 0), 3)
                    cv2.putText(frame,  "U", (x1-10, y1 - 5), 1,
                                2, (0, 255, 0), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame,  "Desconocido", (x1-10, y1 - 5), 1,
                                2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
