import cv2
import numpy as np

def face_pos(box,match_name=[]):
    for i in np.arange(len(box)):
        if not match_name:
            continue
        else:
            profile = match_name[i]
    try:
        return profile
    except UnboundLocalError:
        pass
