# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/FourierTransform.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/outimage.png --frequency_threshold_method frequency --image_band_index 0 --frequency_threshold 30

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import urllib.request
import math
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
ap.add_argument("-j", "--image_band_index", required=True, help="channel index 0, 1, or 2 to use in image")
ap.add_argument("-p", "--frequency_threshold", required=True, help="discard the highest x frequencies in the image e.g. 30")
ap.add_argument("-m", "--frequency_threshold_method", required=True, help="frequency or magnitude")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]
frequency_threshold_method = args["frequency_threshold_method"]
image_band_index = int(args["image_band_index"])
frequency_threshold = int(args["frequency_threshold"])

img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
img_shape = img.shape

if len(img_shape) == 3:
    if img_shape[2] == 3:
        b,g,r = cv2.split(img)
        if image_band_index == 0:
            img = b
        if image_band_index == 1:
            img = g
        if image_band_index == 2:
            img = r

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum 1'), plt.xticks([]), plt.yticks([])
# plt.show()

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
#print(magnitude_spectrum)

# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum 2'), plt.xticks([]), plt.yticks([])
# plt.show()

rows, cols = img.shape
#print(img.shape)
crow,ccol = rows/2 , cols/2
crow = int(round(crow))
ccol = int(round(ccol))

def getMax10(x):
    return np.argpartition(x, -10)[-10:]

if frequency_threshold_method == 'frequency':
    fshift[crow-frequency_threshold:crow+frequency_threshold, ccol-frequency_threshold:ccol+frequency_threshold] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
if frequency_threshold_method == 'magnitude':
    if frequency_threshold == 10:
        mask = np.zeros((rows, cols, 2), np.uint8)
        max10_a1 = np.apply_along_axis(getMax10, axis=1, arr=fshift)
        max10_a2 = np.apply_along_axis(getMax10, axis=0, arr=fshift)

        col_count = 0
        row_count = 0
        total_1 = 0
        total_a = []
        for x in np.nditer(max10_a1):
            total_a.append(x)
            mask[row_count, x] = 1
            col_count += 1
            total_1 += 1
            if col_count == max10_a1.shape[1]:
                col_count = 0
                row_count += 1

        col_count = 0
        row_count = 0
        total_2 = 0
        for x in np.nditer(max10_a2):
            total_a.append(x)
            mask[x, row_count] = 1
            col_count += 1
            total_2 += 1
            if col_count == max10_a2.shape[1]:
                col_count = 0
                row_count += 1

        print(len(np.unique(max10_a1)))
        print(len(np.unique(max10_a2)))
        print(len(np.unique(total_a)))

        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# plt.subplot(131),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.show()
# 
# cv2.waitKey(0)
# print("FT")
# print(img_back.shape)
plt.imsave(outfile_path, img_back, cmap='gray')
