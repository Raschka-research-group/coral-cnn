import imageio
import os
import numpy as np
from mlxtend.image import EyepadAlign
import pyprind

morph_orig_dir = 'morph2-unaligned'
morph_aligned_dir = 'morph2-aligned'

# Get average image
eyepad = EyepadAlign(verbose=1)
eyepad.fit_directory(target_img_dir=morph_orig_dir,
                     target_width=200,
                     target_height=240,
                     file_extension='.JPG')  # note the capital letters


# Center nose of the average image
nose_coord = eyepad.target_landmarks_[33].copy()
disp_vec = np.array([100, 120]) - nose_coord
translated_shape = eyepad.target_landmarks_ + disp_vec

eyepad_centnoise = EyepadAlign(verbose=1)
eyepad_centnoise.fit_values(target_landmarks=translated_shape,
                            target_width=200,
                            target_height=240)


# Align images to centered average image
flist = [f for f in os.listdir(morph_orig_dir) if f.endswith('.JPG')]
pbar = pyprind.ProgBar(len(flist), title='Aligning images ...')

for f in flist:
    pbar.update()
    img = imageio.imread(os.path.join(morph_orig_dir, f))

    img_tr = eyepad.transform(img)
    if img_tr is not None:
        imageio.imsave(os.path.join(morph_aligned_dir, f[:-4]+'.jpg'), img_tr)
