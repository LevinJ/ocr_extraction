import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt



class Findlines(object):
    def __init__(self):
      
        return
    def get_canny_edge(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

#         plt.imshow(gray_show[...,::-1])
#         kernel_size = 5
#         blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
#         
#         blur_gray = (blur_gray * 255).astype(np.uint8)
        
        low_threshold = 150
        high_threshold = 180
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return gray,edges
    def find_lines(self, fname):
        img = cv2.imread(fname)
        
        gray,edges = self.get_canny_edge(img)
        
        return img, gray,edges
    def stack_image_horizontal(self, imgs, max_img_width = None, max_img_height=None):
        return self.__stack_image(imgs, axis = 1, max_img_width = max_img_width, max_img_height=max_img_height)
    def stack_image_vertical(self, imgs, max_img_width = None, max_img_height=None):
        return self.__stack_image(imgs, axis = 0, max_img_width = max_img_width, max_img_height=max_img_height)
    def __stack_image(self, imgs, axis = None, max_img_width = None, max_img_height=None):
        #first let's make sure all the imge has same size
        img_sizes = np.empty([len(imgs), 2], dtype=int)
        for i in range(len(imgs)):
            img = imgs[i]
            img_sizes[i] = np.asarray(img.shape[:2])
        if max_img_width is None:
            max_img_width = img_sizes[:,1].max()
        if max_img_height is None:
            max_img_height = img_sizes[:,0].max()
        for i in range(len(imgs)):
            img = imgs[i]
            img_width = img.shape[1]
            img_height = img.shape[0]
            if (img_width == max_img_width) and (img_height == max_img_height):
                continue
            imgs[i] = cv2.resize(img, (max_img_width,max_img_height))
            
            
        #stack the image
        for i in range(len(imgs)):
            img = imgs[i]
            if len(img.shape) == 2:
                #if this is a binary image or gray image
                if np.max(img) == 1:
                    scaled_img = np.uint8(255*img/np.max(img))
                else:
                    scaled_img = img
                imgs[i] = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)
#                 plt.imshow(imgs[i][...,::-1])
        res_img = np.concatenate(imgs, axis=axis)
        return res_img
        
    def run(self):

        
        fnames = ['./data/ocr/00000012AI20160328023.jpg','./data/ocr/00000015AI20160328023.jpg',
                  './data/ocr/00000021AI20160329001.jpg','./data/ocr/00000026AI20160329003.jpg','./data/ocr/00000031AI20160325010.jpg']
        
        fnames = ['./data/ocr/00000012AI20160328023.jpg','./data/ocr/00000015AI20160328023.jpg',
                  './data/ocr/00000021AI20160329001.jpg']

        res_imgs = []
        for fname in fnames:
            img, gray,edges = self.find_lines(fname)
            res_imgs.append(self.stack_image_horizontal([img, gray,edges]))
           
        
        res_imgs = self.stack_image_vertical(res_imgs)
#         res_imgs = np.array(res_imgs)
#         res_imgs = np.concatenate(res_imgs, axis=0)
        
        
        res_imgs = res_imgs[...,::-1]
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= Findlines()
    obj.run()