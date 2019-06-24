# import the necessary packages
import cv2
import numpy as np

class CropPolygonsToSingleSquareRectangularImage:
    def __init__(self):
        pass

    def crop(self, input_image, polygons):
        pts_array = []
        for polygon in polygons:
            for point in polygon:
                x = point['x']
                y = point['y']

                x = int(round(x))
                y = int(round(y))
                pts_array.append([x,y])

        pts = np.array(pts_array)
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        finalImage = input_image[y:y+h, x:x+w]

        return finalImage
