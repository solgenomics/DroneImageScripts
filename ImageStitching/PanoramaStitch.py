# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageStitching/PanoramaStitch.py --images_urls http://localhost:3000/data/images/image_files/a8/11/e4/76/e7f24fbccc79bed797b73a31/TFV0TzMMlp.png,http://localhost:3000/data/images/image_files/1a/be/f4/eb/580da78f17245b318268df0d/GjUZHIhA25.png,http://localhost:3000/data/images/image_files/ac/31/ab/7e/9f332950a2f1d824a2100b61/4UJdR27u5Q.png,http://localhost:3000/data/images/image_files/63/12/5d/a7/dad19de8898c2e509d7b6bcf/pn2tyXhUz5.png --outfile_path /export/archive/mystitchedimage.png

# import the necessary packages
from Panorama.Panorama import Stitcher
import argparse
import imutils
import cv2
import urllib.request
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_urls", required=True, help="comma separated string of url paths to images")
ap.add_argument("-o", "--outfile_path", required=True, help="complete file path for output stitched image")
args = vars(ap.parse_args())

images_urls = args["images_urls"]
outfile_path = args["outfile_path"]
image_urls = images_urls.split(",")
images = []
for i in image_urls:
    print(i)
    with urllib.request.urlopen(i) as url_response:
        image = np.asarray(bytearray(url_response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #image = cv2.resize(image,(480, 320))
        images.append(image)

print("Starting...")
stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA
stitch_result = stitcher.stitch(images)
result = stitch_result[1]
print("Complete")

# from Panorama.Panorama import Stitcher
# stitcher = Stitcher()
# (kpsA, kpsB, result) = stitcher.stitch(images)

# from PythonMultipleImageStitching.code.pano import Stitch (Relies on images being ordered in a general left to right order)
# s = Stitch(None, images_urls)
# s.leftshift()
# s.rightshift()
# result = s.leftImage

#cv2.imshow("Result", result)
cv2.imwrite(outfile_path, result)
#cv2.imwrite("streakedimage.png", result)
#cv2.waitKey(0)
