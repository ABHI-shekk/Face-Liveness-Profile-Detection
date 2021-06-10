import cv2
import numpy as np

def detect(img, cascade):
    rects,_,confidence = cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels = True)
    if len(rects) == 0:
        return (),()
    rects[:,2:] += rects[:,:2]
    return rects,confidence


def convert_rightbox(img,box_right):
    res = np.array([])
    _,x_max = img.shape
    for box_ in box_right:
        box = np.copy(box_)
        box[0] = x_max-box_[2]
        box[2] = x_max-box_[0]
        if res.size == 0:
            res = np.expand_dims(box,axis=0)
        else:
            res = np.vstack((res,box))
    return res


class detect_face_orientation():
    def __init__(self):
        self.detect_frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.detect_profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    def face_orientation(self,gray):
        # frontal_face
        box_frontal,w_frontal = detect(gray,self.detect_frontal_face)
        if len(box_frontal)==0:
            box_frontal = []
            name_frontal = []
        else:
            name_frontal = len(box_frontal)*["front"]
        # left_face
        box_left, w_left = detect(gray,self.detect_profile_face)
        if len(box_left)==0:
            box_left = []
            name_left = []
        else:
            name_left = len(box_left)*["left"]
        # right_face
        gray_flipped = cv2.flip(gray, 1)
        box_right, w_right = detect(gray_flipped,self.detect_profile_face)
        if len(box_right)==0:
            box_right = []
            name_right = []
        else:
            box_right = convert_rightbox(gray,box_right)
            name_right = len(box_right)*["right"]

        boxes = list(box_frontal)+list(box_left)+list(box_right)
        names = list(name_frontal)+list(name_left)+list(name_right)
        return boxes, names
