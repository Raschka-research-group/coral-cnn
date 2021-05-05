import dlib
import cv2
import os
import numpy as np
from PIL import Image

print(f'DLIB: {dlib.__version__}')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')

# Last tested with
# DLIB: 19.22.0
# NumPy: 1.20.2
# OpenCV: 4.5.2


root_path = './'

orig_path = os.path.join(root_path, 'CACD2000')
out_path = os.path.join(root_path, 'CACD2000-centered')

if not os.path.exists(orig_path):
    raise ValueError(f'Original image path {orig_path} does not exist.')

if not os.path.exists(out_path):
    os.mkdir(out_path)

detector = dlib.get_frontal_face_detector()
keep_picture = []


for picture_name in os.listdir(orig_path):
    img = cv2.imread(os.path.join(orig_path, picture_name))

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

            cv2.imwrite(os.path.join(out_path, picture_name), tmp)
            print(f'Wrote {picture_name}')
            keep_picture.append(picture_name)
        except ValueError:
            print(f'Failed {picture_name}')
            pass
