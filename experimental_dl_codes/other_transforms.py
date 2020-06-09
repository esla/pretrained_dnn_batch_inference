import cv2
import numpy as np
from PIL import Image


class ClaheTransform:
    def __init__(self, clip_limit=2.5, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tileGridSize = tile_grid_size

    # def clahe_transform(bgr_image):
    #     grid_size = 8
    #     clahe_img = cv2.createCLAHE(clipLimit=self, tileGridSize=(grid_size, grid_size))
    #
    #     lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    #     # print(type(lab_image))
    #     lab_planes = cv2.split(lab)
    #     lab_planes[0] = clahe_img.apply(lab_planes[0])
    #     # lab_planes[1] = clahe_img.apply(lab_planes[1])
    #     # lab_planes[2] = clahe_img.apply(lab_planes[2])
    #     lab_img = cv2.merge(lab_planes)
    #     final_bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    #     return final_bgr_img

    def __call__(self, im):
        # #img_y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:,:,0]
        # img_y = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2YCrCb)[:, :, 0]
        # clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        # img_y = clahe.apply(img_y)
        # img_output = img_y.reshape(img_y.shape + (1,))
        # #final_img = Image.fromarray(np.uint8(img_y * 255) , 'L')
        #grid_size = 64
        clahe_img = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tileGridSize)

        lab = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2LAB)
        # print(type(lab_image))
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe_img.apply(lab_planes[0])
        lab_planes[1] = clahe_img.apply(lab_planes[1])
        lab_planes[2] = clahe_img.apply(lab_planes[2])
        lab_img = cv2.merge(lab_planes)
        final_bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        #img_output = self.clahe_transform(np.array(im))
        #cv2.imshow("test", final_bgr_img)
        #cv2.waitKey(0)
        return final_bgr_img
