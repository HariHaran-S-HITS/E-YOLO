import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--write_video', help='Do you want to write the output video', default=False)
parser.add_argument('--prototxt', help='Path to deploy.prototxt', default='deploy.prototxt', required=False)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel',
                    default='hed_pretrained_bsds.caffemodel', required=False)
parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
parser.add_argument('--savefile', help='Specifies the output video path', default='output.mp4', type=str)
args = parser.parse_args()

# Load the model.
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)

# Create a display window
kWinName = 'Holistically-Nested_Edge_Detection'


def image_preproccessing(image, frame):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=7)

    return opening


def edge_detection(frame, inputframe, filename):
    args.height, args.width = frame.shape[1], frame.shape[0]
    inp = cv2.dnn.blobFromImage(cv2.resize(frame, (5000, 5000)),
                                scalefactor=10.0,
                                size=(500, 500),
                                mean=(103.939, 116.779, 123.68, 137.86),
                                swapRB=True, crop=False)

    frame = cv2.resize(frame, (500, 500))
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = 256 * out

    out = cv2.resize(out, (1440, 1440))
    out = out.astype(np.uint8)

    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    image = out.copy()

    thresh = image_preproccessing(image, frame)

    frame = cv2.resize(frame, (1440, 1440))

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    count = 0

    _ = False
    for cnt in contours:
        count = count + 1
        x, y, w, h = cv2.boundingRect(cnt)
        out = cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (x-120, y-120), (x - 80 + w, y - 80 + h), (0, 255, 0), 2)

        try:
            cropped = inputframe[y - 120:y - 80 + h, x - 120:x - 80 + w]
            # breakpoint()
            cv2.imwrite(f"Data/Un-Labelled/Cropped/Un-Clustered/{filename}-{count}.png", cropped)
            _ = True
            cv2.imwrite(f"Data/Un-Labelled/Raw/{filename}-{count}.png", frame)
        except:
            pass

    return _


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Capture one frame after another
        ret, img = cap.read()
        if not ret:
            break

        _ = edge_detection(img, img, "")

        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()
