# -*- coding: utf-8 -*-
"""RYAN_GARCIA"""

#Ryan Garcia Homework 3
#CPSC 4820
import pywt
import pywt.data
import scipy
import numpy as np
import math
import pandas as pd
import cv2  
from google.colab.patches import cv2_imshow
from skimage import io
from PIL import Image 
import matplotlib.pylab as plt
import random

#1-1
PF = cv2.imread('paper-flowers.jpg', cv2.IMREAD_COLOR)
img_1 =cv2.cvtColor(PF, cv2.COLOR_BGR2GRAY)
cv2_imshow(img_1)

#1-2
#Coif Level 1
wavelet = 'coif1'

level = 1
phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=level)

plt.plot(x, phi, label='Scaling function')

plt.plot(x, psi, label='Wavelet function')

plt.title('Coiflets Wavelet (Level 1)')
plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.show()

#1-2
#Haar
wavelet = 'haar'

level = 1
phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=level)
plt.plot(x, phi, label='Scaling function')
plt.plot(x, psi, label='Wavelet function')
plt.title('Haar Wavelet (Level 1)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#1-2
#DB4
wavelet = 'db4'

level = 1

phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=level)
plt.plot(x, phi, label='Scaling function')
plt.plot(x, psi, label='Wavelet function')
plt.title(f'Daubechies 4 Wavelet (Level {level})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#1-3
#Coiflet
PF = cv2.imread('paper-flowers.jpg', cv2.IMREAD_COLOR)
img_1 =cv2.cvtColor(PF, cv2.COLOR_BGR2GRAY)
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs1 = pywt.dwt2(img_1, 'bior1.3')
LL, (LH, HL, HH) = coeffs1
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

#1-3
#Haar
PF = cv2.imread('paper-flowers.jpg', cv2.IMREAD_COLOR)
img_1 =cv2.cvtColor(PF, cv2.COLOR_BGR2GRAY)
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
Haar1 = pywt.dwt2(img_1, 'bior1.3')
LL, (LH, HL, HH) = Haar1
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

#1-3
#DB4
PF = cv2.imread('paper-flowers.jpg', cv2.IMREAD_COLOR)
img_1 =cv2.cvtColor(PF, cv2.COLOR_BGR2GRAY)
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
DB4 = pywt.dwt2(img_1, 'bior1.3')
LL, (LH, HL, HH) = DB4
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

#1-4
#Haar Level 2
PF = cv2.imread('paper-flowers.jpg', cv2.IMREAD_COLOR)
img_1 =cv2.cvtColor(PF, cv2.COLOR_BGR2GRAY)
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
Haar2 = pywt.dwt2(img_1, 'bior1.3')
LL, (LH, HL, HH) = Haar2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

#1-4
#Haar Level 3
PF = cv2.imread('paper-flowers.jpg', cv2.IMREAD_COLOR)
img_1 =cv2.cvtColor(PF, cv2.LCOLOR_BGR2GRAY)
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
Haar3 = pywt.dwt2(img_1, 'bior1.3')
LL, (LH, HL, HH) = Haar3
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

#2-1
LJ = cv2.imread('littlejohn.jpg', cv2.IMREAD_COLOR)
paw = cv2.imread('paw.jpg', cv2.IMREAD_COLOR)

img_2 =cv2.cvtColor(LJ, cv2.COLOR_BGR2GRAY)
template =cv2.cvtColor(paw, cv2.COLOR_BGR2GRAY)
cv2_imshow(img_2)
cv2_imshow(template)

#2-2
LJ = cv2.imread('littlejohn.jpg')
paw = cv2.imread('paw.jpg')

# Convert the image and template to grayscale
img_2 = cv2.cvtColor(LJ, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(paw, cv2.COLOR_BGR2GRAY)
result = cv2.matchTemplate(img_2, template, cv2.TM_CCORR_NORMED)

# Find the location of the maximum correlation value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
w, h = template.shape[::-1]
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_2, top_left, bottom_right, 255, 2)

# Display the result
cv2_imshow(img_2)

#2-2
#b
LJ = cv2.imread('littlejohn.jpg')
paw = cv2.imread('paw.jpg')

# Convert the image and template to grayscale
img_2 = cv2.cvtColor(LJ, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(paw, cv2.COLOR_BGR2GRAY)
result = cv2.matchTemplate(img_2, template, cv2.TM_CCOEFF)

# Find the location of the maximum correlation value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
w, h = template.shape[::-1]
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_2, top_left, bottom_right, 255, 2)

# Display the result
cv2_imshow(img_2)

#2-2
#c
LJ = cv2.imread('littlejohn.jpg')
paw = cv2.imread('paw.jpg')

# Convert the image and template to grayscale
img_2 = cv2.cvtColor(LJ, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(paw, cv2.COLOR_BGR2GRAY)
result = cv2.matchTemplate(img_2, template, cv2.TM_SQDIFF)

# Find the location of the maximum correlation value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
w, h = template.shape[::-1]
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_2, top_left, bottom_right, 255, 2)

# Display the result
cv2_imshow(img_2)

#2-4
LJ = cv2.imread('littlejohn.jpg')
paw = cv2.imread('paw_rotated.JPG')

# Convert the image and template to grayscale
img_2 = cv2.cvtColor(LJ, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(paw, cv2.COLOR_BGR2GRAY)
result = cv2.matchTemplate(img_2, template, cv2.TM_SQDIFF)

# Find the location of the maximum correlation value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
w, h = template.shape[::-1]
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_2, top_left, bottom_right, 255, 2)

# Display the result
cv2_imshow(img_2)

#2-5 a
LJ = cv2.imread('littlejohn.jpg')
paw = cv2.imread('paw.jpg')

# Convert the image and template to grayscale
img_2 = cv2.cvtColor(LJ, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(paw, cv2.COLOR_BGR2GRAY)

gauss = np.random.normal(0,.1,img_2.size)
gauss = gauss.reshape(img_2.shape[0],img_2.shape[1])
# Multiply noise to image pixel values
Noisy_image = img_2 + img_2 * gauss
Noisy_image = Noisy_image.astype(np.uint8)
#cv2_imshow(Noisy_image)

result = cv2.matchTemplate(Noisy_image, template, cv2.TM_CCORR_NORMED)

# Find the location of the maximum correlation value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
w, h = template.shape[::-1]
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(Noisy_image, top_left, bottom_right, 255, 2)

# Display the result
cv2_imshow(Noisy_image)

#2-5 b
LJ = cv2.imread('littlejohn.jpg')
paw = cv2.imread('paw.jpg')

# Convert the image and template to grayscale
img_2 = cv2.cvtColor(LJ, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(paw, cv2.COLOR_BGR2GRAY)

gauss = np.random.normal(0,.08,img_2.size)
gauss = gauss.reshape(img_2.shape[0],img_2.shape[1])
# Multiply noise to image pixel values
Noisy_image = img_2 + img_2 * gauss
Noisy_image = Noisy_image.astype(np.uint8)
#cv2_imshow(Noisy_image)

result = cv2.matchTemplate(Noisy_image, template, cv2.TM_CCOEFF)

# Find the location of the maximum correlation value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
w, h = template.shape[::-1]
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(Noisy_image, top_left, bottom_right, 255, 2)

# Display the result
cv2_imshow(Noisy_image)

#2-5 c
LJ = cv2.imread('littlejohn.jpg')
paw = cv2.imread('paw.jpg')

# Convert the image and template to grayscale
img_2 = cv2.cvtColor(LJ, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(paw, cv2.COLOR_BGR2GRAY)

gauss = np.random.normal(0,.11,img_2.size)
gauss = gauss.reshape(img_2.shape[0],img_2.shape[1])
# Multiply noise to image pixel values
Noisy_image = img_2 + img_2 * gauss
Noisy_image = Noisy_image.astype(np.uint8)
#cv2_imshow(Noisy_image)

result = cv2.matchTemplate(Noisy_image, template, cv2.TM_SQDIFF)

# Find the location of the maximum correlation value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
w, h = template.shape[::-1]
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(Noisy_image, top_left, bottom_right, 255, 2)

# Display the result
cv2_imshow(Noisy_image)