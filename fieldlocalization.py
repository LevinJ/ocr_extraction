import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from exploreimages import ExploreImages
from findlines import Findlines



class FieldLocation(Findlines):
    def __init__(self):
        Findlines.__init__(self)
      
        return
    def format_coord(self, x, y):
        pt = self.hsv_img[int(y), int(x), :]
        return 'HSV value, x={:.0f}, y={:.0f}  [h={}, s={}, v={}]'.format(x, y, pt[0],pt[1],pt[2])
    def show_pixel_values(self):
        #for the purpose of showing hsv value
        fname = './data/ocr/00000012AI20160328023.jpg'
        img, gray,edges,hough_lines = self.find_lines(fname)
        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.hsv_img = hsv_img 
        _, ax = plt.subplots()
        ax.format_coord = self.format_coord
        ax.imshow(img[...,::-1])
        plt.show()
        return
    
    def run(self):
        return self.show_pixel_values()
        
        fnames = ['./data/ocr/00000012AI20160328023.jpg']

        
        fnames = ['./data/ocr/00000012AI20160328023.jpg','./data/ocr/00000015AI20160328023.jpg',
                  './data/ocr/00000021AI20160329001.jpg','./data/ocr/00000026AI20160329003.jpg','./data/ocr/00000031AI20160325010.jpg']
         
#         fnames = ['./data/ocr/00000012AI20160328023.jpg','./data/ocr/00000015AI20160328023.jpg',
#                   './data/ocr/00000021AI20160329001.jpg']

        res_imgs = []
        for fname in fnames:
            img, gray,edges,hough_lines = self.find_lines(fname)
            res_imgs.append(self.stack_image_horizontal([img]))
           
        
        res_imgs = self.stack_image_vertical(res_imgs)
#         res_imgs = np.array(res_imgs)
#         res_imgs = np.concatenate(res_imgs, axis=0)
        
        
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        
        plt.show()
        return



if __name__ == "__main__":   
    obj= FieldLocation()
    obj.run()