#!/usr/bin/env python3

from vaegan.models import create_models, build_graph

e,dec,dis = create_models()
q = build_graph(e,dec,dis)

e.load_weights('encoder.090.h5')
dec.load_weights('decoder.090.h5')

vae = q[-2]


from PIL import Image
import numpy as np
import cv2

img = cv2.imread('100627-255.jpg')
img = cv2.resize(img, (64, 64))
img = (img - 127.5) / 127.5
img = img.reshape(1, 64, 64, 3)

def norm(img):
    img = (img + 1.) / 2.
    return img


import matplotlib.pyplot as plt
import numpy as np

recon = vae.predict(img)
z = np.random.normal(size=(1, 128))

new = dec.predict(z)

plt.subplot(131)
plt.imshow(norm(img).squeeze())

plt.subplot(132)
plt.imshow(norm(recon).squeeze())

plt.subplot(133)
plt.imshow(norm(new).squeeze())

plt.show()
