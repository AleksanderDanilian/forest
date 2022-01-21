import os
import cv2
import pandas as pd
import PIL
from PIL import Image
import numpy as np
from helper_functions import plate_detector, visualize

# путь к фото лесовоза, которое будет передано для расчета
IMG_DIR = '/content/drive/MyDrive/AIU/Лесозаготовка/train/2.jpg'

# путь к весам yolo
WEIGHTS_YOLOV5 = '/content/drive/MyDrive/AIU/Лесозаготовка/веса_Yolo_Timber/best.pt'

# путь к весам классификации
WEIGHTS_CLASS = '/content/drive/MyDrive/AIU/Лесозаготовка/Веса_классификация/best.pt'

# путь к результатам распознавания леса (стандартная папка yolo)

# os.makedirs('/content/yolov5/runs/detect')
DETECT_DIR = '/content/yolov5/runs/detect'

# уверенность (объекты ниже этого порога не будут детектированы)
CONF = 0.6

# площадь российского номерного знака
PLATE_AREA = 5.2 * 1.1  # дм^2

pi = 3.14159


def predict_timber(weights_yolov5=WEIGHTS_YOLOV5, img_dir=IMG_DIR, path=DETECT_DIR, conf=CONF, bbox_type='ellipse', final_wide=800, show=1):
  ''' Основная функция. Находит и рассчитывает площадь номера. 
  Предсказывает координаты и размер бревен, рассчитывает площадь бревен.
  на вход: 
  img_dir - путь к изображению
  weighns_yolov5 - обученные веса
  conf - уверенность
  bbox_type - тип bounding box, варианты: ellipse, circle, box
  final_wide - ширина выводимого изображения
  show - визуализировать результат (boolean)

  на выходе:
  resized - изображение шириной final_wide
  bbox_list - список bounding_box (относительные координаты центра, высоты/ширины: x, y, w, h)
  plate - координаты углов номера
  textArr - распознанный номер
  '''

  # запуск предсказание yolo
  os.system(f"python detect.py --weights {weights_yolov5} --img 640 --save-txt --conf {conf} --source {img_dir}")
    
  # определение папки с последним предсказанием yolo в папке /content/yolov5/runs/detect
  detect_dir = max([os.path.join(path, dir) for dir in os.listdir(path)], key=os.path.getctime)

  # чтение предсказанных координат
  bbox_array = np.loadtxt(detect_dir+'/labels/' + os.listdir(detect_dir+'/labels/')[0])

  image = cv2.imread(img_dir)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  bboxes = bbox_array[:,1:] 

  # функция распознавания номера
  plate, textArr = plate_detector(image)
    
  # меняем изображение до подготовленных размеров
  r = float(final_wide) / image.shape[1]
  dim = (final_wide, int(image.shape[0] * r))
  resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)  

  if len(plate):
    plate[:,:,0] = plate[:,:,0] * resized.shape[0]
    plate[:,:,1] = plate[:,:,1] * resized.shape[1]

    pts = np.array(plate, np.int32)

    resized = cv2.polylines(resized, pts, True, (0, 0, 255), thickness=2)

  if len(plate):
    area = Polygon(plate[0]).area # площадь номера в пикселях
  else:
    area = 4000 # средний размер номера

  img, areas = visualize(resized, bboxes, area=area, bbox_type='ellipse')

  d_list = list(map(lambda x:round((x/pi)**0.5*20,2), areas)) #диаметры бревен (2*10)
  s_overall = round(sum(areas)/100, 2) #дм2 в м2
  areas = list(map(lambda x:round(x,2), areas))
  
  img_edited = Image.fromarray(img, 'RGB')
  img_edited.save(detect_dir + f'/{s_overall}_{textArr}.png') # change save_dir later!
  
  bb_pandas = [str(i) for i in bboxes] # по другому список не положить в 1 колонку в пандас

  data = np.vstack((d_list, areas, bb_pandas)).transpose()
  
  df = pd.DataFrame(data = data,
                    columns = ['diameter, cm', 'area, dm2', 'bbox'])
  df.to_csv(detect_dir + f'/{s_overall}_{textArr}.csv') # change save_dir later!

  return resized, bboxes, plate, textArr, df, img_edited
