# importar bibliotecas
from os.path import join, dirname

import numpy as np
import argparse
import cv2

#constantes
PROTOTXT = join(dirname(__file__), "deploy.prototxt.txt")
MODEL = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
INPUT = join(dirname(__file__), "profile.jpg")
CONFIDENCE_THRESHOLD = 0.8

# carregar modelo Caffe
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# carregar imagem
image = cv2.imread(INPUT)

# pegar as dimensões da imagem
(h, w) = image.shape[:2]

# redimeensionar image (300x300)
resized = cv2.resize(image, (300, 300))

# blob network
blob = cv2.dnn.blobFromImage(resized, 1.0, (300, 300), (104.0, 177.0, 123.0))

# passar blob pela rede
net.setInput(blob)

# detections
detections = net.forward()

# iterar sobre todas as detecções
for i in range(0, detections.shape[2]):
    # extrair confiança de cada previsão
    confidence = detections[0, 0, i, 2]

    if confidence > CONFIDENCE_THRESHOLD:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # desenhar retângulo
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)

face = image[startY:endY, startX:endX]
blured = cv2.GaussianBlur(face, (int(w / 3.0), int(h / 3.0)), 0)

censurada = image.copy()
censurada[startY:endY, startX:endX] = blured

# exibir imagem final
cv2.imshow("Face Detection", censurada)
cv2.waitKey(0)



