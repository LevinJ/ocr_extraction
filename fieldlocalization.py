import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from exploreimages import ExploreImages
from findlines import Findlines
from detect_peaks import detect_peaks



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
        fname = './data/ocr/00000030AI20160329003.jpg'
        img, gray,edges,hough_lines = self.find_lines(fname)
        
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2HLS)
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2HSV)
        channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2Lab)
        self.hsv_img = channels 
        _, ax = plt.subplots()
        ax.format_coord = self.format_coord
        ax.imshow(img[...,::-1])
        plt.show()
        return
    def show_channels(self, fname):
#         fname = './data/ocr/00000026AI20160325020.jpg'
#         fname = './data/ocr/00000030AI20160329003.jpg'
        img, gray,edges,hough_lines = self.find_lines(fname)
        
        channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2Lab)
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2HSV)
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2HLS)
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2Lab) #looks very good< L
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2YUV)
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2RGB)
#         plt.imshow(channels[:,:,1],cmap='gray')
        self.res = self.stack_image_vertical([channels[:,:,0],channels[:,:,1],channels[:,:,2]])
        cv2.line(hough_lines,(0,0),(1900,0),(255,0,0),10)

        self.hough_lines = hough_lines
        self.gray = gray

        return
    def remove_edge_noise(self):
        return
    def find_number(self, region):

        field = region[:, 1380:]
        
        #downsample and use it for processing
        self.rgb = cv2.pyrDown(field);
        self.small = cv2.cvtColor(self.rgb,  cv2.COLOR_BGR2GRAY);

        
        #morphological gradient
        morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));
        self.grad = cv2.morphologyEx(self.small, cv2.MORPH_GRADIENT, morphKernel);
        
        #binarize
    
        thres, self.bw = cv2.threshold(self.grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU);
        
        #Remvoe noise around edge
        self.erosion = self.bw.copy()
        pad = 5
        self.erosion = self.bw.copy()
        self.erosion[:,0:(pad)] = 0
        self.erosion[:,-pad:] = 0
        self.erosion[0:(pad), :] = 0
        self.erosion[-(pad):, :] = 0
        
        
        #connect horizontally oriented regions
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1));
        self.connected = cv2.morphologyEx(self.erosion, cv2.MORPH_CLOSE, morphKernel);
#         self.connected = self.erosion.copy()
        
#         self.find_rec(self.erosion)
#          
#          
#          
#         #find contours
        im2, contours, hierarchy = cv2.findContours(self.connected.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
          
          
        self.res = self.rgb.copy()
        mask = np.zeros(self.rgb.shape[:2], np.uint8)
        self.contour = self.rgb.copy()
          
        # filter contours
        x_top = []
        y_top = []
        x_bottom = []
        y_bottom = []
        for idx in range(0, len(contours)):
            x, y, rect_width, rect_height = cv2.boundingRect(contours[idx])
            if rect_height < 20 or rect_width < 20:
                continue
            
            cv2.drawContours(mask, contours, idx, 255, cv2.FILLED)
            cv2.drawContours(self.contour, contours, idx, (0,0,255))
            
            x_top.append(x)
            y_top.append(y)
            x_bottom.append(x + rect_width)
            y_bottom.append(y + rect_height)
       
        
        if len(x_top) != 0:  
            x_top, y_top, x_bottom, y_bottom = self.adjust_rect(x_top[-1], y_top[-1], x_bottom[-1], y_bottom[-1], pad, self.rgb.shape[:2])
            cv2.rectangle(self.res, (x_top, y_top ), (x_bottom, y_bottom ), (0,255,0),1)
          
          

        return  self.res
    def find_rec(self, erosion):
#         plt.imshow(erosion, cmap='gray')
        #find border in x direction
        x_sum = erosion.sum(axis = 0)/255.0
        x_sum[x_sum<10] = 0
#         print("{}".format(x_sum))
        left,right = x_sum.argmin(), x_sum.argmax()
        
        y_sum = erosion.sum(axis = 1)/255.0
        y_sum[y_sum<10] = 0
#         x_sum[x_sum<10] = 0
        print("{}".format(y_sum))
        top,bottom = y_sum.argmin(), y_sum.argmax()
        
        
        #remove noise
#         found_peaks = detect_peaks(x_sum, mph=20, mpd=50)
#         self.hist_img = self.__get_histogram_img(erosion, x_sum, found_peaks,prefix_str="x_sum")
        
        #find border in x direction
        
        #remove noise
      
        return 
   
    def __get_histogram_img(self, img, histogram, found_peaks, prefix_str=""):
        img_height = img.shape[0]
        hist_img = img.copy()
        
        hist_img = cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)
#         hist_img = np.uint8(255*hist_img/np.max(hist_img))
        
        
        width = len(histogram)
#         
        histogram = histogram/float(histogram.max()) * img_height
        ys = -histogram[:, np.newaxis] + img_height
#         ys = histogram[:, np.newaxis]
        xs = np.arange(0, width)[:, np.newaxis]
        pts = np.concatenate([xs,ys], axis = 1).astype(np.int32)
        cv2.polylines(hist_img, [pts], color=(255,0,0),isClosed=False,thickness=1)
        if len(found_peaks) != 0:
            peaks_info = prefix_str + '{}'.format(found_peaks)
            cv2.putText(hist_img,peaks_info,(100,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),4)
        for peak in found_peaks:
            pt1 = (peak, img_height)
            pt2 = (peak, int(img_height/2))
            cv2.line(hist_img, pt1,pt2,(0,0,255),thickness=3)
        
        return hist_img
    def find_sex(self, region):
        
        field = region[:, 460:550]
        
        #downsample and use it for processing
        self.rgb = cv2.pyrDown(field);
        self.small = cv2.cvtColor(self.rgb,  cv2.COLOR_BGR2GRAY);
        
        #morphological gradient
        morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));
        self.grad = cv2.morphologyEx(self.small, cv2.MORPH_GRADIENT, morphKernel);
        
        #binarize
    
        thres, self.bw = cv2.threshold(self.grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU);
        
        #Remvoe noise around edge
        self.erosion = self.bw.copy()
        pad = 5
        self.erosion = self.bw.copy()
        self.erosion[:,0:(pad)] = 0
#         self.erosion[:,-pad:] = 0
        self.erosion[0:(pad), :] = 0
        self.erosion[-(pad):, :] = 0
        
        #connect horizontally oriented regions
        self.connected = self.erosion.copy()
#         #find contours
        im2, contours, hierarchy = cv2.findContours(self.connected.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
          
          
        self.res = self.rgb.copy()
        mask = np.zeros(self.rgb.shape[:2], np.uint8)
        self.contour = self.rgb.copy()
          
        # filter contours
        max_rect_area = None
        max_x = 0
        max_y=0
        max_rect_width = 0
        max_rect_height = 0
        for idx in range(0, len(contours)):
            x, y, rect_width, rect_height = cv2.boundingRect(contours[idx])
            cv2.drawContours(self.contour, contours, idx, (0,0,255))
#             cv2.rectangle(self.contour, (x, y), (x + rect_width, y + rect_height), (0,255,0),1)
            
            if rect_height < 22 or rect_width < 8:
                continue    
            if x< 10 and rect_width < 10:
                continue     
            
            
            cv2.drawContours(mask, contours, idx, 255, cv2.FILLED)
           
            if(max_rect_area is None or rect_width*rect_height > max_rect_area):
                max_rect_area = rect_width*rect_height
                max_x = x
                max_y = y
                max_rect_width = rect_width
                max_rect_height = rect_height
        
        if max_rect_area is not None:
            x_top, y_top, x_bottom, y_bottom = self.adjust_rect(max_x, max_y, 
                                                                max_x+max_rect_width, max_y+max_rect_height, pad, self.rgb.shape[:2])     
            cv2.rectangle(self.res, (x_top, y_top), (x_bottom, y_bottom), (0,255,0),1)
          
          
        self.field_name = field
         
        return self.res
    def find_type(self, region):

        field = region[:, 640:1230]
        
        #downsample and use it for processing
        self.rgb = cv2.pyrDown(field);
        self.small = cv2.cvtColor(self.rgb,  cv2.COLOR_BGR2GRAY);
#         self.small = cv2.Sobel(self.small, cv2.CV_8U, 1, 0)
#         channels = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2Lab)
#         self.small = channels[:,:,0]
        #self.small = cv2.cvtColor(self.rgb,  cv2.COLOR_BGR2GRAY);
        
#         kernel = np.ones((3,3),np.uint8)
#         self.small = cv2.erode(self.small,kernel,iterations = 1)
        
        #morphological gradient
        morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));
        self.grad = cv2.morphologyEx(self.small, cv2.MORPH_GRADIENT, morphKernel);
        
        #binarize
    
        thres, self.bw = cv2.threshold(self.grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU);
        
        #Remvoe noise around edge
        self.erosion = self.bw.copy()
        pad = 7
        self.erosion = self.bw.copy()
        self.erosion[:,0:(pad)] = 0
        self.erosion[:,-pad:] = 0
        self.erosion[0:(pad), :] = 0
        self.erosion[-(pad):, :] = 0
        
        
        #connect horizontally oriented regions
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1));
        self.connected = cv2.morphologyEx(self.erosion, cv2.MORPH_CLOSE, morphKernel);
        
#         self.find_rec(self.erosion)
#          
#          
#          
#         #find contours
        im2, contours, hierarchy = cv2.findContours(self.connected.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
          
          
        self.res = self.rgb.copy()
        mask = np.zeros(self.rgb.shape[:2], np.uint8)
        self.contour = self.rgb.copy()
          
        # filter contours
        x_top = []
        y_top = []
        x_bottom = []
        y_bottom = []
        for idx in range(0, len(contours)):
            x, y, rect_width, rect_height = cv2.boundingRect(contours[idx])
            if rect_height < 20 or rect_width < 3:
                continue
            if (rect_width < 20 and x < 15):
                continue
            cv2.drawContours(mask, contours, idx, 255, cv2.FILLED)
            cv2.drawContours(self.contour, contours, idx, (0,0,255))
            
            x_top.append(x)
            y_top.append(y)
            x_bottom.append(x + rect_width)
            y_bottom.append(y + rect_height)
        
        if len(x_top) != 0:  
            x_top, y_top, x_bottom, y_bottom = self.adjust_rect(np.min(x_top), np.min(y_top), np.max(x_bottom), np.max(y_bottom), pad, self.rgb.shape[:2])
            cv2.rectangle(self.res, (x_top, y_top ), (x_bottom, y_bottom ), (0,255,0),1)
          
          
         
        return self.res
    def adjust_rect(self, x_top, y_top, x_bottom, y_bottom, pad, shape):
        height, width = shape
        if x_top == pad:
            x_top = 0
        if y_top == pad:
            y_top = 0
        if x_bottom == width - pad:
            x_bottom = width
        if y_bottom == height -pad:
            y_bottom = height
        return x_top, y_top, x_bottom, y_bottom
    def find_names(self, region):
        field_pos = 380
        field = self.fields[:, 65:380]
        
        #downsample and use it for processing
        self.rgb = cv2.pyrDown(field);
        self.small = cv2.cvtColor(self.rgb,  cv2.COLOR_BGR2GRAY);
#         self.small = cv2.Sobel(self.small, cv2.CV_8U, 1, 0)
#         channels = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2Lab)
#         self.small = channels[:,:,0]
        #self.small = cv2.cvtColor(self.rgb,  cv2.COLOR_BGR2GRAY);
        
#         kernel = np.ones((3,3),np.uint8)
#         self.small = cv2.erode(self.small,kernel,iterations = 1)
        
        #morphological gradient
        morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));
        self.grad = cv2.morphologyEx(self.small, cv2.MORPH_GRADIENT, morphKernel);
        
        #binarize
    
        thres, self.bw = cv2.threshold(self.grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU);
        
        #Remvoe noise around edge
        self.erosion = self.bw.copy()
        pad = 7
        self.erosion = self.bw.copy()
        self.erosion[:,0:(pad)] = 0
        self.erosion[:,-pad:] = 0
        self.erosion[0:(pad), :] = 0
        self.erosion[-(pad):, :] = 0
        
        #connect horizontally oriented regions
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1));
        self.connected = cv2.morphologyEx(self.erosion, cv2.MORPH_CLOSE, morphKernel);
#         #find contours
        im2, contours, hierarchy = cv2.findContours(self.connected.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
          
          
        self.res = self.rgb.copy()
        mask = np.zeros(self.rgb.shape[:2], np.uint8)
        self.contour = self.rgb.copy()
        
          
        # filter contours
        x_top = []
        y_top = []
        x_bottom = []
        y_bottom = []
        for idx in range(0, len(contours)):
            x, y, rect_width, rect_height = cv2.boundingRect(contours[idx])
            cv2.drawContours(self.contour, contours, idx, (0,0,255))
            if rect_height < 15 or rect_width < 10:
                continue
            if x< 10 and rect_width < 20:
                continue
            cv2.drawContours(mask, contours, idx, 255, cv2.FILLED)
            
            maskroi = mask[y:y+rect_height, x:x+rect_width]
            x_top.append(x)
            y_top.append(y)
            x_bottom.append(x + rect_width)
            y_bottom.append(y + rect_height)
        if len(x_top) != 0:  
            x_top, y_top, x_bottom, y_bottom = self.adjust_rect(x_top[-1], y_top[-1], x_bottom[-1], y_bottom[-1], pad, self.rgb.shape[:2])
            cv2.rectangle(self.res, (x_top, y_top ), (x_bottom, y_bottom ), (0,255,0),1)
           
            
          
          
         
        return self.res
    def find_fields(self, fname, raise_exception = True):
        img, gray,edges,hough_lines = self.find_lines(fname, raise_exception = raise_exception)
        
        self.fields = hough_lines.copy()
        
        self.hough_lines = hough_lines
        name = self.find_names(hough_lines.copy())
        sex = self.find_sex(hough_lines.copy())
        type = self.find_type(hough_lines.copy())
        number = self.find_number(hough_lines.copy())
        
#         name_pos = 380
#         sex_pos = 560
#         type_pos = 1230
        
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2Lab)
#         self.thresholded = cv2.inRange(channels, (0,0,0), (160,255,255))
        
#         channels = cv2.cvtColor(hough_lines, cv2.COLOR_BGR2HLS)
#         self.thresholded = cv2.inRange(channels, (0,0,0), (255,160,50))
#         
#         
#         cv2.line(self.fields,(name_pos,0),(name_pos,119),(255,0,0),10)
#         cv2.line(self.fields,(sex_pos,0),(sex_pos,119),(255,0,0),10)
#         cv2.line(self.fields,(type_pos,0),(type_pos,119),(255,0,0),10)
        return name,sex,type,number,self.hough_lines
    def save_all_fields(self):
        expl = ExploreImages()
        fnames = expl.get_all_imags()
        region_folder = './data/fields/number/'
        if not os.path.exists(region_folder):
            os.makedirs(region_folder)
        count = 0
        for fname in fnames:
            count += 1
            print("{}, {}".format(count, fname))
            self.find_fields(fname)
            file_path = region_folder + os.path.basename(fname)
            plt.imsave(file_path, self.res[...,::-1])
            
            
        return
    def save_all_fields_per_image(self):
        expl = ExploreImages()
        fnames = expl.get_all_imags()
        count = 0
        for fname in fnames:
            count += 1
            print("{}, {}".format(count, fname))
            region_folder = './data/result/' + os.path.basename(fname)[:-4] + '/'
            if not os.path.exists(region_folder):
                os.makedirs(region_folder)
            field_names = ['name','sex', 'type', 'number', 'region']
            res = self.find_fields(fname)
            for name, img in zip(field_names, res):
                file_path = region_folder + name + '.jpg'
                plt.imsave(file_path, img[...,::-1])
            
            
        return
    
    def run(self):
        return self.save_all_fields_per_image()
        
#         return self.save_all_fields()
        
        fnames = ['./data/ocr/00000026AI20160325020.jpg']

        
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
#             self.show_channels(fname)
            res_imgs.append(self.stack_image_horizontal([self.res]))
#             res_imgs.append(self.stack_image_horizontal([self.small,  self.grad, self.bw, self.erosion,self.connected, self.contour, self.res]))
           
        
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