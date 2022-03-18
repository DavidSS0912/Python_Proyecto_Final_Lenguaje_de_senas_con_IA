import cv2
import mediapipe as mp
import os

nombre = "Letra_O"
direccion = "C:/Users/David/Documents/Semestre_2021_2/Python/PF/Validacion"
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print("Carpeta creada: ", carpeta)
    os.makedirs(carpeta)

cont = 0

cap = cv2.VideoCapture(0)

clase_manos = mp.solutions.hands
manos = clase_manos.Hands()

dibujo = mp.solutions.drawing_utils

while(True):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):

                alto, ancho, c = frame.shape
                cX, cY = int(lm.x * ancho), int(lm.y * alto)

                posiciones.append([id, cX, cY])
                dibujo.draw_landmarks(
                    frame, mano, clase_manos.HAND_CONNECTIONS)

            if len(posiciones) != 0:
                p1 = posiciones[4]
                p2 = posiciones[20]
                p3 = posiciones[12]
                p4 = posiciones[0]
                p5 = posiciones[9]

                x1, y1 = (p5[1] - 100), (p5[2] - 150)
                ancho, alto = (x1 + 100), (y1 + 150)
                x2, y2 = x1 + ancho, y1 + alto+100

                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, '{}/300'.format(cont), (x1-10, y1 - 5), 1,
                            2, (0, 255, 0), 3, cv2.LINE_AA)
            dedos_reg = cv2.resize(dedos_reg, (100, 100),
                                   interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(carpeta + "/Dedos_{}.jpg".format(cont), dedos_reg)
            cont += 1

    cv2.imshow("Abecedario", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break
cap.release()
cv2.destroyAllWindows()
