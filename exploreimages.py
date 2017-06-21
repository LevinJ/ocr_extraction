import sys
import os
# from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


class ExploreImages(object):
    def __init__(self):
        return
    def get_img_size(self):
        file_pattern = './data/ocr/*.jpg'
        fnames = self.get_all_imags(file_pattern)
        img_shapes = []
        for fname in fnames:
            img = cv2.imread(fname)
            print("{}: {}".format(fname , img.shape))
            img_shapes.append(img.shape)
        img_shapes = np.array(img_shapes)
        print("height width ratio {}".format(np.array(img_shapes[:,0]/img_shapes[:,1])))
        print("img average size {}".format(img_shapes.mean(axis = 0)))
        return
    
    def get_all_imags(self, file_pattern='./data/ocr/*.jpg'):
       
        
        fnames = glob.glob(file_pattern)
        fnames.sort()
        return fnames
    
    def run(self):
        self.get_img_size()
       
        return



if __name__ == "__main__":   
    obj= ExploreImages()
    obj.run()
    
    
    
# ./data/ocr/00000008AI20160329042.jpg: (1502, 2472, 3)
# ./data/ocr/00000012AI20160328023.jpg: (1499, 2468, 3)
# ./data/ocr/00000012AI20160329004.jpg: (1497, 2348, 3)
# ./data/ocr/00000013AI20160324012.jpg: (1504, 2528, 3)
# ./data/ocr/00000013AI20160328023.jpg: (1501, 2464, 3)
# ./data/ocr/00000013AI20160329001.jpg: (1500, 2604, 3)
# ./data/ocr/00000013AI20160329004.jpg: (1504, 2216, 3)
# ./data/ocr/00000014AI20160324002.jpg: (1503, 2452, 3)
# ./data/ocr/00000014AI20160324012.jpg: (1505, 2524, 3)
# ./data/ocr/00000014AI20160325028.jpg: (1000, 1644, 3)
# ./data/ocr/00000014AI20160325031.jpg: (985, 1620, 3)
# ./data/ocr/00000014AI20160328023.jpg: (1500, 2468, 3)
# ./data/ocr/00000014AI20160329003.jpg: (982, 1652, 3)
# ./data/ocr/00000015AI20160127014.jpg: (996, 1648, 3)
# ./data/ocr/00000015AI20160324002.jpg: (1505, 2452, 3)
# ./data/ocr/00000015AI20160328023.jpg: (1497, 2476, 3)
# ./data/ocr/00000015AI20160408009.jpg: (994, 1664, 3)
# ./data/ocr/00000016AI20160127014.jpg: (991, 1664, 3)
# ./data/ocr/00000016AI20160325010.jpg: (1509, 2428, 3)
# ./data/ocr/00000016AI20160328023.jpg: (1501, 2468, 3)
# ./data/ocr/00000016AI20160408009.jpg: (977, 1656, 3)
# ./data/ocr/00000017AI20160328023.jpg: (1497, 2480, 3)
# ./data/ocr/00000018AI20160324012.jpg: (1510, 2464, 3)
# ./data/ocr/00000020AI20160325020.jpg: (992, 1648, 3)
# ./data/ocr/00000020AI20160329003.jpg: (987, 1656, 3)
# ./data/ocr/00000021AI20160324020.jpg: (986, 1652, 3)
# ./data/ocr/00000021AI20160329001.jpg: (1515, 2480, 3)
# ./data/ocr/00000021AI20160329003.jpg: (989, 1660, 3)
# ./data/ocr/00000022AI20160329004.jpg: (1502, 2476, 3)
# ./data/ocr/00000025AI20160324012.jpg: (1516, 2472, 3)
# ./data/ocr/00000025AI20160325010.jpg: (1507, 2464, 3)
# ./data/ocr/00000025AI20160328023.jpg: (1497, 2460, 3)
# ./data/ocr/00000025AI20160329003.jpg: (990, 1660, 3)
# ./data/ocr/00000026AI20160325020.jpg: (1016, 1656, 3)
# ./data/ocr/00000026AI20160328023.jpg: (1503, 2460, 3)
# ./data/ocr/00000026AI20160329003.jpg: (986, 1644, 3)
# ./data/ocr/00000027AI20160325020.jpg: (1009, 1660, 3)
# ./data/ocr/00000027AI20160328023.jpg: (1512, 2460, 3)
# ./data/ocr/00000028AI20160325020.jpg: (976, 1644, 3)
# ./data/ocr/00000028AI20160328023.jpg: (1502, 2464, 3)
# ./data/ocr/00000029AI20160324018.jpg: (978, 1648, 3)
# ./data/ocr/00000029AI20160328023.jpg: (1500, 2464, 3)
# ./data/ocr/00000030AI20160324018.jpg: (991, 1656, 3)
# ./data/ocr/00000030AI20160325020.jpg: (980, 1648, 3)
# ./data/ocr/00000030AI20160328023.jpg: (1499, 2468, 3)
# ./data/ocr/00000030AI20160329003.jpg: (985, 1596, 3)
# ./data/ocr/00000030AI20160329004.jpg: (1498, 2464, 3)
# ./data/ocr/00000031AI20160324018.jpg: (987, 1652, 3)
# ./data/ocr/00000031AI20160324020.jpg: (981, 1648, 3)
# ./data/ocr/00000031AI20160325010.jpg: (1504, 2404, 3)
# height width ratio [ 0.60760518  0.60737439  0.63756388  0.59493671  0.60917208  0.57603687
#   0.67870036  0.612969    0.59627575  0.60827251  0.60802469  0.60777958
#   0.59443099  0.60436893  0.61378467  0.6046042   0.59735577  0.59555288
#   0.62149918  0.60818476  0.58997585  0.60362903  0.61282468  0.60194175
#   0.59601449  0.5968523   0.6108871   0.59578313  0.60662359  0.61326861
#   0.61160714  0.60853659  0.59638554  0.61352657  0.61097561  0.59975669
#   0.60783133  0.61463415  0.59367397  0.60957792  0.5934466   0.60876623
#   0.59842995  0.59466019  0.60737439  0.61716792  0.60795455  0.59745763
#   0.59526699  0.62562396]
# img average size [ 1276.94  2102.48     3.  ]
