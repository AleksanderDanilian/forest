import cv2
import numpy as np
import os 
import pandas as pd
import PIL
from PIL import Image

import cv2

import numpy as np                               # Библиотека работы с массивами
import pandas as pd
import matplotlib.pyplot as plt                  # Для отрисовки графиков
from matplotlib.path import Path

# from IPython.display import Image
import PIL
from PIL import Image

import random                                    # Для генерации случайных чисел 
import os                                        # Для работы с файлами 
import subprocess
import sys

import timeit
import time 

from shapely.geometry import Polygon            # Для функции площади многоугольника

# Функции визуализации 

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (40, 0, 0) # White
pi = 3.1415926535
PLATE_AREA = 5.2 * 1.1

def visualize_bbox(img, bbox, area=0, color=BOX_COLOR, thickness=2, bbox_type='ellipse'):
    x_cntr, y_cntr, w, h = map(int, bbox) # координаты центров и размеры рамок 
    x_min, x_max, y_min, y_max = int(x_cntr - w/2), int(x_cntr + w/2), int(y_cntr - h/2), int(y_cntr + h/2)
    center_coordinates = (x_cntr, y_cntr)

    if bbox_type=='ellipse':
      a, b = int(w/2), int(h/2)
      axesLength = (a, b)
      cv2.ellipse(img, center_coordinates, axesLength, angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)  #angle=0, startAngle=0, endAngle=360, 
    elif bbox_type=='circle':
      radius = int(sq ** 0.5 / 2) # sq?
      cv2.circle(img, center_coordinates, radius, color=color, thickness=thickness)
      pass
    else:
      cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)


    # площадь бревна в квадратных дециметрах (делим площадь бревна в пикселях на площадь номера, потом умноражем на размер номера)
    # на бревнах указываем диаметр бревна
    area_float = (w/2)*(h/2)*pi/area*PLATE_AREA # площадь бревна до округления (дм2)
    area = str(round(area_float,1))
    diameter = str(int(round((area_float/pi)**0.5*20,0))) # 20 = 2*10 (10 - перевод в см, 2 - вынесли за скобки sqrt)

    fsc = 0.8 
    ((text_width, text_height), _) = cv2.getTextSize(diameter, cv2.FONT_HERSHEY_SIMPLEX, fsc, 1)    
    cv2.putText(
        img,
        text=diameter,
        org=(x_cntr - int(text_width/2), y_cntr + int(0.5 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fsc, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img, area_float  # возвращает изображение с отметкой бревна и площадь бревна 


def visualize(image, bboxes, area=0, bbox_type='ellipse'):
    img = image.copy()
    bbxs = bboxes.copy()
    area_list = []  # список площадей каждого бревна

    # print(img.shape, len(bbxs))
    if len(bbxs):
      bbxs[:,[0,2]] = bbxs[:,[0,2]] * img.shape[1]
      bbxs[:,[1,3]] = bbxs[:,[1,3]] * img.shape[0]
    for bbox in bbxs:
        img, timb_area = visualize_bbox(img, bbox, area=area, bbox_type=bbox_type)
        area_list.append(timb_area)
    return img, area_list


# Функция распознавания номера
# на вход: изображение
# на выход: координаты углов и результат распознавания номера 

# Import license plate recognition tools.
from NomeroffNet.YoloV5Detector import Detector
detector = Detector()
detector.load()

from NomeroffNet.BBoxNpPoints import NpPointsCraft, getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
npPointsCraft = NpPointsCraft()
npPointsCraft.load()

from NomeroffNet.OptionsDetector import OptionsDetector
from NomeroffNet.TextDetector import TextDetector

from NomeroffNet import TextDetector
from NomeroffNet import textPostprocessing

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

textDetector = TextDetector.get_static_module("ru")
textDetector.load("latest")

def plate_detector(img):

  targetBoxes = detector.detect_bbox(img)
  all_points = np.array(npPointsCraft.detect(img, targetBoxes,[5,2,0])) # координаты номера

  if len(all_points):
    all_points[:,:,0] = all_points[:,:,0] / img.shape[0]
    all_points[:,:,1] = all_points[:,:,1] / img.shape[1]
    
  # cut zones
  zones = convertCvZonesRGBtoBGR([getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points])

  # predict zones attributes
  regionIds, countLines = optionsDetector.predict(zones)
  regionNames = optionsDetector.getRegionLabels(regionIds)

  # find text with postprocessing by standart
  textArr = textDetector.predict(zones)
  textArr = textPostprocessing(textArr, regionNames)

  return all_points, textArr

