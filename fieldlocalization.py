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
        
        channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2HSV)
        channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2Lab)
        self.hsv_img = channels 
        _, ax = plt.subplots()
        ax.format_coord = self.format_coord
        ax.imshow(img[...,::-1])
        plt.show()
        return
    def show_channels(self):
        fname = './data/ocr/00000012AI20160328023.jpg'
        fname = './data/ocr/00000030AI20160329003.jpg'
        img, gray,edges,hough_lines = self.find_lines(fname)
        
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2HSV)
#         channels = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2Lab) #looks very good< L
#         channels = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#         channels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.imshow(channels[:,:,1],cmap='gray')
        channels_img = self.stack_image_vertical([channels[:,:,0],channels[:,:,1],channels[:,:,2]])
        plt.imshow(channels_img[...,::-1])
        plt.show()
        return
    def find_fields(self, fname, raise_exception = True):
        img, gray,edges,hough_lines = self.find_lines(fname, raise_exception = raise_exception)
        
        self.fields = hough_lines.copy()
        name_pos = 380
        sex_pos = 560
        type_pos = 1230
        
        
        cv2.line(self.fields,(name_pos,0),(name_pos,119),(255,0,0),10)
        cv2.line(self.fields,(sex_pos,0),(sex_pos,119),(255,0,0),10)
        cv2.line(self.fields,(type_pos,0),(type_pos,119),(255,0,0),10)
        return
    
    def run(self):
        self.show_pixel_values()
        self.show_channels()
#         return self.save_all_regions()
        
        fnames = ['./data/ocr/00000030AI20160329003.jpg']

        
#         fnames = ['./data/ocr/00000012AI20160328023.jpg','./data/ocr/00000015AI20160328023.jpg',
#                   './data/ocr/00000015AI20160127014.jpg','./data/ocr/00000026AI20160329003.jpg',
#                   './data/ocr/00000031AI20160325010.jpg','./data/ocr/00000030AI20160329003.jpg',
#                   './data/ocr/00000026AI20160325020.jpg']
#         fnames = ['./data/ocr/00000030AI20160329003.jpg']
         
#         fnames = ['./data/ocr/00000012AI20160328023.jpg','./data/ocr/00000015AI20160328023.jpg',
#                   './data/ocr/00000021AI20160329001.jpg']

        res_imgs = []
        for fname in fnames:
            print("image {}".format(fname))
            self.find_fields(fname, raise_exception = False)
            res_imgs.append(self.stack_image_horizontal([self.fields]))
           
        
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