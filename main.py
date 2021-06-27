# importar pacotes
import os

from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import time
from os.path import dirname, join
import os

from filters import grayscale, original, sketch, sepia, blur, canny

# constantes
PROTOTXT = join(dirname(__file__), "deploy.prototxt.txt")
MODEL = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
CONFIDENCE_THRESHOLD = 0.7
WEBCAM = os.environ.get('WEBCAM', 1)

# carregar o modelo
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# iniciar o streaming
def main():

    print('[INFO] starting video stream')
    vs = VideoStream(src=WEBCAM).start()
    time.sleep(2.0)

    filters = {
        '0': original,
        '1': grayscale,
        '2': sketch,
        '3': sepia,
        '4': blur,
        '5': canny,
        '6': None,
        '7': None
    }

    print("""Press any of the following keys to:
        0: Original Image
        1: Grayscale
        2: Sketch
        3: Sepia
        4: Blur
        5: Canny
        6: Face detection
        7: Blur face
        q: Quit""")

    # initial_filter
    selected_filter = '0'

    while True:
        # ler frames
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        filter = filters.get(selected_filter)
        if filter is not None:
            frame = filter(frame)

        # iterar ao longo das deteccoes
        for i in range(0, detections.shape[2]):
            # exemplo de intervalo de confianca
            confidence = detections[0, 0, i, 2]

            # selecionar apenas intervalos acima do threshold
            if confidence > CONFIDENCE_THRESHOLD:
                # label da confiança
                text = "{:.2f}%".format(confidence * 100)

                # calcular o bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                if selected_filter == '6':
                    frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)

                if selected_filter == '7':
                    face = frame[startY:endY, startX:endX]
                    blured = cv2.GaussianBlur(face, (33, 33), 0)
                    frame[startY:endY, startX:endX] = blured

        # exibir frame na tela
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key in [ord(k) for k in filters.keys()]:
            selected_filter = chr(key)

    # destruir as janelas e interromper o stream
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
