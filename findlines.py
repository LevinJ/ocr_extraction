import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from exploreimages import ExploreImages



class Findlines(object):
    def __init__(self):
      
        return
    def crop_roi(self, edges, vertices):
        #next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(edges)   
        ignore_mask_color = 255   
        
        #this time we are defining a four sided polygon to mask
        imshape = edges.shape
#         vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
#         vertices = np.array([[(10,539),(460,290), (480,290), (930,539)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges
    def get_hough_lines(self, edges):
        rho = 1
        theta = np.pi/180
        threshold = 10
        min_line_length = 100
        max_line_gap = 5
        
        color_edges = np.dstack((edges, edges, edges)) 
        line_image = np.copy(color_edges)*0 #creating a blank to draw lines on
        
       
#         masked_edges = self.crop_roi(edges)
        #run Hough on edge detected image

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        #iterate over the output "lines" and draw lines on the blank
        eps = 1e-2
        for line in lines:
            for x1,y1,x2,y2 in line:
                if math.fabs((x2 -x1)) < eps:
                    continue
                k = (y2-y1)/float((x2-x1))
                if math.fabs(k)> eps:
                    continue
                dist = math.fabs(x2-x1)
                print("slope {}, distance {}".format(k, dist))
                #retain only the horiztal lines
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                
        #creating a "color" binary image to combine with line image     
        #drawing the lines on the edge image
        combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
        plt.imshow(combo)
        
        return combo
    def get_canny_edge(self, gray):
        # Convert to grayscale

#         plt.imshow(gray_show[...,::-1])
#         kernel_size = 5
#         blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
#         
#         blur_gray = (blur_gray * 255).astype(np.uint8)
        
        low_threshold = 150
        high_threshold = 180
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges
    def find_lines(self, fname):
        img = cv2.imread(fname)
        img = cv2.resize(img, (2100,1276))
        
        
        
        
        #crop region of interest
        
#         vertices = np.array([[(210,250),(1880,210), (1880,315), (210,315)]], dtype=np.int32)
#         img = self.crop_roi(img, vertices)
        img = img[250:316, 210:1880]
        gray = edges =hough_lines = img
#         plt.imshow(img[...,::-1])
        
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 
#         
#         edges = self.get_canny_edge(gray)
#         
#         hough_lines = self.get_hough_lines(edges)
        
        return img, gray,edges,hough_lines
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
    def save_all_regions(self):
        expl = ExploreImages()
        fnames = expl.get_all_imags()
        region_folder = './data/regions/'
        if not os.path.exists(region_folder):
            os.makedirs(region_folder)
        for fname in fnames:
            img, gray,edges,hough_lines = self.find_lines(fname)
            file_path = region_folder + os.path.basename(fname)
            plt.imsave(file_path, img[...,::-1])
            
            
        return
        
    def run(self):
        return self.save_all_regions()
        
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
    obj= Findlines()
    obj.run()