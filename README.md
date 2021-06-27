# Realtime Filters and Face Detection

In this project we are using preprocessing image techniques to create some instagram-like
filters that are applied in real time by using a connected cam.

The filters available are:
1. Grayscale
2. Sketch
3. Sepia
4. Blur
5. Canny
6. Face detection (prints a green square around the face)
7. Blur face


## Usage

- Clone the repo
- Configure WEBCAM variable (through environment or code) to use the correct cam index
- Run application
- Apply filters using keys [0-7]


```shell
git clone https://github.com/virb30/realtime_face_filters.git
# export WEBCAM=1 
set WEBCAM=1
python main.py
```