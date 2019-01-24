import dlib
import cv2
import os
import numpy as np
from PIL import Image

root_path = './'

detector = dlib.get_frontal_face_detector()
keep_picture = []

for picture_name in os.listdir(root_path + 'CACD2000/'):
    img = cv2.imread(root_path + 'CACD2000/' + picture_name)

    detected = detector(img, 1)

    if len(detected) != 1:  # skip if there are 0 or more than 1 face
        continue

    for idx, face in enumerate(detected):
        width = face.right() - face.left()
        height = face.bottom() - face.top()
        tol = 15
        up_down = 5
        diff = height-width

        if(diff > 0):
            if not diff % 2:  # symmetric
                tmp = img[(face.top()-tol-up_down):(face.bottom()+tol-up_down),
                          (face.left()-tol-int(diff/2)):(face.right()+tol+int(diff/2)),
                          :]
            else:
                tmp = img[(face.top()-tol-up_down):(face.bottom()+tol-up_down),
                          (face.left()-tol-int((diff-1)/2)):(face.right()+tol+int((diff+1)/2)),
                          :]
        if(diff <= 0):
            if not diff % 2:  # symmetric
                tmp = img[(face.top()-tol-int(diff/2)-up_down):(face.bottom()+tol+int(diff/2)-up_down),
                          (face.left()-tol):(face.right()+tol),
                          :]
            else:
                tmp = img[(face.top()-tol-int((diff-1)/2)-up_down):(face.bottom()+tol+int((diff+1)/2)-up_down),
                          (face.left()-tol):(face.right()+tol),
                          :]

        try:
            tmp = np.array(Image.fromarray(np.uint8(tmp)).resize((120, 120), Image.ANTIALIAS))

            cv2.imwrite(root_path + 'CACD2000-centered/' + picture_name, tmp)
            keep_picture.append(picture_name)
        except ValueError:
            pass
