# import the necessary packages
import cv2
import numpy as np

class CropPolygonsToSingleImage:
    def __init__(self):
        pass

    def crop(self, input_image, polygons):
        input_image_size = input_image.shape
        original_y = input_image_size[0]
        original_x = input_image_size[1]
        minY = original_y
        minX = original_x
        maxX = -1
        maxY = -1

        for polygon in polygons:
            for point in polygon:
                x = point['x']
                y = point['y']

                x = int(round(x))
                y = int(round(y))
                point['x'] = x
                point['y'] = y

                if x < minX:
                    minX = x
                if x > maxX:
                    maxX = x
                if y < minY:
                    minY = y
                if y > maxY:
                    maxY = y

        cropedImage = np.zeros_like(input_image)
        for y in range(0,original_y):
            for x in range(0, original_x):

                if x < minX or x > maxX or y < minY or y > maxY:
                    continue

                for polygon in polygons:
                    polygon_mat = []
                    for p in polygon:
                        polygon_mat.append([p['x'], p['y']])

                    if cv2.pointPolygonTest(np.asarray([polygon_mat]),(x,y),False) >= 0:
                        if len(input_image_size) == 3:
                            for j in range(input_image_size[2]):
                                cropedImage[y, x, j] = input_image[y, x, j]
                        else:
                            cropedImage[y, x] = input_image[y, x]

        # Now we can crop again just the envloping rectangle
        finalImage = cropedImage[minY:maxY,minX:maxX]

        return finalImage
