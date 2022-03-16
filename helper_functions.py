import math
import re

import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path

import PIL
from PIL import Image, ImageOps, ImageEnhance

import random
import os
import subprocess
import sys

import timeit
import time

from shapely.geometry import Polygon

# from GPSPhoto import gpsphoto # какая-то ошибка лезет при импорте

# Функции визуализации 

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (40, 0, 0)  # White
pi = 3.14159
PLATE_AREA = 5.2 * 1.1


def visualize_bbox(img, bbox, area=0, color=BOX_COLOR, thickness=2, bbox_type='ellipse'):
    """
    Функция подсчета общей площади бревен
    :param img - изображение (numpy array)
    :param bbox - выход YOLO, bounding box бревна в абсолютных величинах (пример [ 311.88, 80.549, 26.25, 23.728])
    :param area - площадь номерного знака в пикселях
    :param color - цвет рамки bbox
    :param thickness - толщина рамки
    :param bbox_type - тип рамки

    :return
    img - изображение с отметкой бревна (numpy array)
    timb_area - площадь бревна
    scale_sq - масштаб изображения
    """

    x_cntr, y_cntr, w, h = map(int, bbox)  # координаты центров и размеры рамок
    x_min, x_max, y_min, y_max = int(x_cntr - w / 2), int(x_cntr + w / 2), int(y_cntr - h / 2), int(y_cntr + h / 2)
    center_coordinates = (x_cntr, y_cntr)

    if bbox_type == 'ellipse':
        a, b = int(w / 2), int(h / 2)
        axes_length = (a, b)
        cv2.ellipse(img, center_coordinates, axes_length, angle=0, startAngle=0, endAngle=360, color=color,
                    thickness=thickness)  # angle=0, startAngle=0, endAngle=360,
    elif bbox_type == 'circle':
        radius = int(sq ** 0.5 / 2)
        cv2.circle(img, center_coordinates, radius, color=color, thickness=thickness)
        pass
    else:
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    # площадь бревна в кв. дециметрах (делим площадь бревна в пикселях на площадь номера, потом умножаем
    # на размер номера)
    # на бревнах указываем диаметр бревна
    timb_area = (w / 2) * (h / 2) * pi / area * PLATE_AREA  # площадь бревна до округления (дм2)
    scale_sq = PLATE_AREA / area  # масштаб фото дм2/pixel2
    diameter = str(
        int(round((timb_area / pi) ** 0.5 * 20, 0)))  # 10 - перевод в см, 2 - вынесли за скобки sqrt

    fsc = 0.8
    ((text_width, text_height), _) = cv2.getTextSize(diameter, cv2.FONT_HERSHEY_SIMPLEX, fsc, 1)
    cv2.putText(
        img,
        text=diameter,
        org=(x_cntr - int(text_width / 2), y_cntr + int(0.5 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fsc,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img, timb_area, scale_sq


def visualize(image, bboxes, area=0, bbox_type='ellipse'):
    """
    Функция подсчета общей площади бревен

    :param image - изображение (numpy array)
    :param bboxes - выход YOLO, список координат бревен в относительных координатах
    :param area - площадь номерного знака в пикселях
    :param bbox_type - тип рамки

    :return
    img - размеченное изображение (numpy array)
    area_list - список площадей бревен
    scale_sq - масштаб изображения
    """

    img = image.copy()
    bbxs = bboxes.copy()
    area_list = []  # список площадей каждого бревна

    # перевод bboxes из относительных в абсолютные величины.
    if len(bbxs):
        bbxs[:, [0, 2]] = bbxs[:, [0, 2]] * img.shape[1]
        bbxs[:, [1, 3]] = bbxs[:, [1, 3]] * img.shape[0]
    for bbox in bbxs:
        img, timb_area, scale_sq = visualize_bbox(img, bbox, area=area, bbox_type=bbox_type)
        area_list.append(timb_area)
    return img, area_list, scale_sq


# Import license plate recognition tools.

os.chdir('/content/forest/nomeroff-net-master')

from NomeroffNet.YoloV5Detector import Detector
from NomeroffNet.BBoxNpPoints import NpPointsCraft, getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
from NomeroffNet.OptionsDetector import OptionsDetector
from NomeroffNet.TextDetector import TextDetector
from NomeroffNet import TextDetector
from NomeroffNet import textPostprocessing

detector = Detector()
detector.load()
npPointsCraft = NpPointsCraft()
npPointsCraft.load()
optionsDetector = OptionsDetector()
optionsDetector.load("latest")
textDetector = TextDetector.get_static_module("ru")
textDetector.load("latest")

os.chdir('/content/forest')


def plate_detector(img):
    """
    Функция распознавания номера
    :param img - изображение

    :return
    all_points - координаты углов
    text_arr - результат распознавания номера (str)
    """

    target_boxes = detector.detect_bbox(img)
    all_points = np.array(npPointsCraft.detect(img, target_boxes, [5, 2, 0]))  # координаты номера

    if len(all_points):
        all_points[:, :, 0] = all_points[:, :, 0] / img.shape[0]
        all_points[:, :, 1] = all_points[:, :, 1] / img.shape[1]

    # cut zones
    zones = convertCvZonesRGBtoBGR([getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points])

    # predict zones attributes
    region_ids, count_lines = optionsDetector.predict(zones)
    region_names = optionsDetector.getRegionLabels(region_ids)

    # find text with postprocessing by standard
    text_arr = textDetector.predict(zones)
    text_arr = textPostprocessing(text_arr, region_names)

    return all_points, text_arr


def crop_center(pil_img, crop_width: int, crop_height: int) -> Image:
    """
    Функция для обрезки изображения по центру.

    :param pil_img - изображение, открытое через PIL
    :param crop_width - ширина обрезки (кол-во пикселей, которое хотим срезать вдоль оси x)
    :param crop_height - высота обрезки (кол-во пикселей, которое хотим срезать вдоль оси y)

    :return
    обрезанное изображение
    """

    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def thumbnail(pil_img, dims):
    """
    Функция для обрезки изображения по наибольшему размеру с срхранением пропорций.

    :param pil_img - изображение, открытое через PIL
    :param dims - требуемые размеры итогового изображения

    :return
    img - квадратное изображение размером (dims, dims) без искажения исходного отношения высоты и ширины.
    Пустоты залиты черным цветом.
    """

    img_w, img_h = pil_img.size[0], pil_img.size[1]
    max_val = np.argmax([img_w, img_h])
    scale = pil_img.size[max_val] / dims

    if img_w > img_h:
        new_img_w = dims
        new_img_h = int(img_h / scale)
    else:
        new_img_h = dims
        new_img_w = int(img_w / scale)

    pil_img = pil_img.resize((new_img_w, new_img_h))
    img = crop_center(pil_img, dims, dims)

    return img


def prepare_crops(img_dir, bbox, save_path, file_name,
                  resize_dim=(64, 64), res_type='zoom', normalize=False):
    """
    функция подготавливает вырезанные bbox для подачи в модель по классификации изображений
    и сохраняет нарезки изображений в папку.

    :param img_dir - путь, где хранится изображение
    :param bbox - bounding box - выход YOLO для одного бревна
    :param save_path - путь для сохранения результатов обрезанных изрбражений
    :param file_name - имя файла для сохранения
    :param resize_dim - размеры обрезки изображения перед подачей в нейронку
    :param res_type - тип обрезки изображения (str - 'stretch', 'zoom' или 'thumbnail')
    :param normalize - нормализация изображения (boolean)

    :return
    resized_image - изображение, подготовленное для подачи в нейронку по сравнению бревен (numpy array)
    """

    image = Image.open(img_dir)
    width, height = image.size
    xc, yc, w, h = bbox

    crop_box = [int(xc * width - w * width / 2), int(yc * height - h * height / 2), int(xc * width + w * width / 2),
                int(yc * height + h * height / 2)]
    cropped_image = image.crop(crop_box)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cropped_image.save(os.path.join(save_path, file_name))

    if res_type == 'stretch':
        resized_image = cropped_image.resize(resize_dim)
    elif res_type == 'zoom':
        resized_image = ImageOps.fit(cropped_image, resize_dim, Image.ANTIALIAS)
    elif res_type == 'thumbnail':
        resized_image = thumbnail(cropped_image, resize_dim[0])
    resized_image = np.array(resized_image)
    if normalize:
        resized_image = resized_image / 255

    return resized_image


def calc_stack_geometry(bboxes, scale_sq, img_dir):
    """
    Функция находит геометрические размеры штабеля
    с древесиной.
    :param bboxes - bounding boxes - выход YOLO
    :param scale_sq - масштаб изображения (дм/pixel)**2
    :param img_dir - директория, где хранится изображение

    :return
    stack_width - ширина штабеля в см
    stack_height - высота штабеля в см
    x_min, y_min, x_max, y_max - координаты рамки, ограничивающей штабель
    width, height - размеры изображения в пикселях

    """
    image = Image.open(img_dir)
    width, height = image.size

    x_min, y_min, x_max, y_max = width, height, 0, 0
    for i in range(len(bboxes)):
        xc, yc, w, h = bboxes[i]
        crop_box = [int(xc * width - w * width / 2), int(yc * height - h * height / 2), int(xc * width + w * width / 2),
                    int(yc * height + h * height / 2)]
        if crop_box[0] < x_min:  # левый верхний угол
            x_min = crop_box[0]
        if crop_box[1] < y_min:  # левый верхний угол
            y_min = crop_box[1]
        if crop_box[2] > x_max:  # правый нижний угол
            x_max = crop_box[2]
        if crop_box[3] > y_max:  # правый нижний угол
            y_max = crop_box[3]

    scale = 10 * math.sqrt(scale_sq)  # линейный масштаб cм/pixel (был дм2/pixel2)

    stack_height = (y_max - y_min) * scale
    stack_width = (x_max - x_min) * scale

    return stack_width, stack_height, x_min, y_min, x_max, y_max, width, height


def get_GPS(img_dir):
    """
    Функция извлекает геоданные из фотографии, если они там есть.
    :param img_dir - директория с изображением

    :return
    gps_coords - координаты gps - широта и долгота
    """
    data = gpsphoto.getGPSData(img_dir)
    if data != {}:
        gps_coords = {'Широта': data['Latitude'], 'Долгота': data['Longitude']}
    else:
        gps_coords = 'У фотографии нет гео метки'
    return gps_coords


def bboxes_to_int(img, bboxes, PIL=False):
    """
    Функция переводит координаты bounding boxes из относительных величин
    в абсолютные.
    :param img: изображение (numpy array или PIL)
    :param bboxes: bounding boxes - выход YOLO
    :param PIL: если открывали изображение через PIL, то PIL = True

    :return:
    x_cntr, y_cntr, w, h - координаты центра рамки, ширина и высота рамки
    """
    if PIL:
        height, width = img.size[1], img.size[0]
    else:
        height, width = img.shape[0], img.shape[1]
    x_cntr, y_cntr, w, h = bboxes
    x_cntr, w = int(x_cntr * width), int(w * width)
    y_cntr, h = int(y_cntr * height), int(h * height)

    return x_cntr, y_cntr, w, h


def draw_classes(img, bboxes, w_class_list, detect_dir, color=(255, 0, 0), text_color=(255, 255, 0), thickness=2):
    """
    Функция вырисовки классов бревен на изображении. Сохраняет изображение wood_classes.png с классами в detect_dir.

    :param img: изображение, открытрое через PIL или cv2
    :param bboxes: bounding boxes - выход YOLO
    :param w_class_list: список классифицированных бревен (соотнесен по порядку с нумерацией bboxes)
    :param detect_dir: расположение папки, в которую будем сохранять файл с классификацией бревен
    :param color: цвет рамки
    :param text_color: цвет текста внутри рамки
    :param thickness: толщина рамки
    """

    for i in range(len(bboxes)):
        x_cntr, y_cntr, w, h = bboxes_to_int(img, bboxes[i], PIL=False)

        center_coordinates = (int(x_cntr), int(y_cntr))

        a, b = int(w / 2), int(h / 2)
        axes_length = (a, b)

        cv2.ellipse(img, center_coordinates, axes_length, angle=0, startAngle=0, endAngle=360, color=color,
                    thickness=thickness)

        fsc = 0.8

        ((text_width, text_height), _) = cv2.getTextSize(str(w_class_list[i]), cv2.FONT_HERSHEY_SIMPLEX, fsc, 1)
        cv2.putText(
            img,
            text=str(w_class_list[i]),
            org=(x_cntr - int(text_width / 2), y_cntr + int(0.5 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fsc,
            color=text_color,
            lineType=cv2.LINE_AA,
        )

    img_edited = Image.fromarray(img, 'RGB')
    img_edited.save(detect_dir + f'/wood_classes.png')


def find_nearest(arr, value, ret='value', amt=3):
    """
    Поиск элемента в массиве `arr` ближайшего к скаляру `value`.

    :param arr: массив, в котором осуществляется поиск
    :param value: значение, близкое к которому мы будем искать в массиве arr
    :param ret: что возвращает функция - value(str: value) или index (str: idx) элемента в массиве arr
    :param amt: кол-во элементов, которые заберем из массива arr, наиболее близких к value

    :return:
    idx - индекс элемента массива arr, наиболее близкого к value
    arr[idx] - значение элемента массива arr, наиболее близкого к value
    """

    diff_arr = np.abs(arr - value)
    sorted_arr = sorted(list(np.abs(diff_arr)))

    idx = [list(diff_arr).index(sorted_arr[i]) for i in range(amt)]

    if ret == 'value':
        return arr[idx]
    elif ret == 'idx':
        return idx


def compare_tables(df_1, df_2, margin=0.05):
    """
    Функция возвращает словарь, в котором сопоставляются древесина с 1й картинки(датафрейма df_1), древесине со 2й
    картинки(датафрейма df_2).

    :param df_1: Датафрейм первой картинки
    :param df_2: Датафрейм второй картинки
    :param margin: диапазон поиска бревен на 1й картинке в процентах от значения площади конкретного бревна на
    2й картинке

    :return: словарь, сопоставляющий бревна 1й и 2й картинок
    """

    areas_list_1 = df_1['area, dm2'].values
    classes_list_1 = df_1['wood class'].values

    areas_list_2 = df_2['area, dm2'].values
    classes_list_2 = df_2['wood class'].values

    matching_dict = {}

    for num, el in enumerate(areas_list_2):
        idx = np.argwhere((areas_list_1 > el * (1 - margin)) & (areas_list_1 < el * (1 + margin)))
        idx = idx[classes_list_2[num] == classes_list_1[idx]]  # оставляем полено, которое подходит по классу
        s_nearest = find_nearest(areas_list_1[idx],
                                 areas_list_2[num])  # если более 1го полена, то берем то, которое ближе по площади
        idx = np.argwhere(areas_list_1 == s_nearest)[0][0]  # ищем индекс нашей площади в начальном списке

        matching_dict[num] = idx

    return matching_dict


def draw_ellipses(img, bboxes, color_box, color_text, thickness, num):
    """
    Функция рисует рамки(эллипсы) на картинках.

    :param img: изображение в формате Numpy array
    :param bboxes: bounding boxes - выход YOLO
    :param color_box: цвет рамок
    :param color_text: цвет текста внутри рамки
    :param thickness: толщина рамок
    :param num: номер бревна
    """

    x_cntr, y_cntr, w, h = bboxes_to_int(img, bboxes, PIL=False)

    center_coordinates = (x_cntr, y_cntr)

    a, b = int(w / 2), int(h / 2)
    axes_length = (a, b)

    cv2.ellipse(img, center_coordinates, axes_length, angle=0, startAngle=0, endAngle=360, color=color_box,
                thickness=thickness)

    fsc = 0.8
    ((text_width, text_height), _) = cv2.getTextSize(str(num), cv2.FONT_HERSHEY_SIMPLEX, fsc, 1)
    cv2.putText(
        img,
        text=f'{num}',
        org=(x_cntr - int(text_width / 2), y_cntr + int(0.5 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fsc,
        color=color_text,
        lineType=cv2.LINE_AA
    )


def draw_matching_bbox(img_dir_1, img_dir_2, df_1, df_2, matching_dict, color_box=(0, 0, 255),
                       color_text=(255, 255, 255), thickness=2):
    """
    Функция рисования номеров бревен на 1 и 2 картинках.

    :param img_dir_1: директория 1го изображения
    :param img_dir_2: директория 2го изображения
    :param df_1: первый датафрейм (выход функции predict_timber() для 1й картинки)
    :param df_2: второй датафрейм (выход функции predict_timber() для 2й картинки)
    :param matching_dict: словарь, сопоставляющий бревна с первой картинки бревнам со второй (согласно их порядку в df)
    :param color_box: цвет рамок
    :param color_text: цвет текста
    :param thickness: толщина рамок
    :return:
    img_1 - 1е изображение без изменений (после predict_timber())
    img_2 - 2е изображение с нумерацией бревен, соответсвующее определенным в ходе работы НС парам бревен.
    percentage_same - процент бревен 1й картинки, найденных на второй картинке
    """

    bboxes_1 = get_bbox_from_df(df_1['bbox'].values)
    bboxes_2 = get_bbox_from_df(df_2['bbox'].values)

    img_1 = cv2.imread(img_dir_1)
    img_2 = cv2.imread(img_dir_2)

    drawn_woods_2pic = []
    count = 0

    for i in range(len(bboxes_1)):
        draw_ellipses(img_1, bboxes_1[i], color_box, color_text, thickness, num=i)

        if matching_dict[i] != 'нет_совпадений':

            try:
                idx_same_wood = matching_dict[i]
                if idx_same_wood not in drawn_woods_2pic:
                    draw_ellipses(img_2, bboxes_2[idx_same_wood], color_box, color_text, thickness, num=i)
                    count += 1
                    drawn_woods_2pic.append(
                        idx_same_wood)  # костыль, чтобы не рисовать бревна, назначенные дважды. Исправить позже.
            except KeyError:
                print('Не нашли подходящего бревна')
                continue

    percentage_same = round(100 * count / len(df_1), 2)

    return img_1, img_2, percentage_same


def get_bbox_from_df(df_bbox_values):
    """
    Функция получения bboxes в числовом формате из pandas df.

    :param df_bbox_values: df['bbox'].values

    :return:
    bboxes - list of bbox coordinates
    """
    bboxes = []

    for el in df_bbox_values:
        el = el.strip('[]').strip()
        el = re.findall(r'0\.\d{0,5}', el)
        bboxes.append(list(map(float, el)))

    return bboxes


def compare_images(img_dir_1, img_dir_2, df_1, df_2, model_path, dim, acc_margin):
    """
    Функция сравнения изображений лесовоза.

    :param img_dir_1: директория первого изображения для операции сравнения
    :param img_dir_2: директория второго изображения для операции сравнения
    :param df_1: первый датафрейм (выход функции predict_timber() для 1й картинки)
    :param df_2: второй датафрейм (выход функции predict_timber() для 2й картинки)
    :param model_path: директория, в которой хранятся веса модели для сравнения картинок
    :param dim: размеры картинки(каждого бревна) для сравнения
    :param acc_margin: точность, ниже которой выход нейронки будет считаться недействительным (нейронка не нашла
    совпадений по бревну с 1 картинки у бревен со второй картинки)

    :return:
    match_dict - словарь, сопоставляющий бревна с первой картинки бревнам со второй (согласно их порядку в df)
    """

    model = load_model(model_path)

    img_1 = Image.open(img_dir_1)
    img_2 = Image.open(img_dir_2)

    bboxes_1 = get_bbox_from_df(df_1['bbox'].values)
    bboxes_2 = get_bbox_from_df(df_2['bbox'].values)

    crops_1 = []
    crops_2 = []
    match_dict = {}

    for i in range(len(df_1)):
        match = []

        x_cntr, y_cntr, w, h = bboxes_to_int(img_1, bboxes_1[i], PIL=True)
        x_min, x_max, y_min, y_max = int(x_cntr - w / 2), int(x_cntr + w / 2), int(y_cntr - h / 2), int(y_cntr + h / 2)

        # img_crop_1 = img_1[y_min:y_max, x_min:x_max] # height, width // cv2 crop
        img_crop_1 = img_1.crop((x_min, y_min, x_max, y_max))
        img_crop_1 = ImageOps.fit(img_crop_1, dim, Image.ANTIALIAS)
        crops_1.append(img_crop_1)

        img_crop_1 = np.array(img_crop_1)

        if i == 0:
            for j in range(len(df_2)):
                x_cntr, y_cntr, w, h = bboxes_to_int(img_2, bboxes_2[j], PIL=True)
                x_min, x_max, y_min, y_max = int(x_cntr - w / 2), int(x_cntr + w / 2), int(y_cntr - h / 2), int(
                    y_cntr + h / 2)

                img_crop_2 = img_2.crop((x_min, y_min, x_max, y_max))

                img_crop_2 = ImageOps.fit(img_crop_2, dim, Image.ANTIALIAS)
                crops_2.append(img_crop_2)

        for j in range(len(df_2)):
            img_crop_2 = np.array(crops_2[j])

            result = model.predict([np.expand_dims(img_crop_1, 0), np.expand_dims(img_crop_2, 0)])
            match.extend(result)

        top_3 = [sorted(match)[-k] for k in range(1, 4)]

        possible_match_idx = [match.index(top_3[i]) for i in range(len(top_3)) if
                              top_3[i] > acc_margin]  # берем топ 3 совпадения по выходу нейронки c acc > acc_margin)

        possible_areas = find_nearest(df_2['area, dm2'].values, df_1['area, dm2'][i], ret='idx', amt=3)
        if len(possible_match_idx) > 0:

            for m in range(len(possible_match_idx)):
                # проверяем, есть ли среди ближ. знач площадей из df_2 те же индексы, что и при проверке на соответствие картинок
                if possible_match_idx[m] in possible_areas:
                    idx_win = possible_match_idx[m]
                    break
                else:
                    idx_win = None
            if idx_win == None:
                p_areas = np.array([df_2['area, dm2'][idx] for idx in possible_match_idx])
                idx_win = possible_match_idx[list(p_areas).index(
                    find_nearest(p_areas, df_1['area, dm2'][i], ret='value',
                                 amt=1))]  # ищем ближайшие площади к искомой среди топа выхода нейронки
        else:
            idx_win = 'нет_совпадений'

        match_dict[i] = idx_win

    return match_dict


def get_train_batch(path, dim, im_id, crops_1):
    """
    Функция создания train batch для обучения нейронок для каждого бревна.
    :param path:
    :param dim:
    :param im_id:
    :param crops_1:
    :return:
    """

    x_tr_1 = []
    x_tr_2 = []
    y_tr = []

    if path is not None:
        image_names = os.listdir(path)
        iter_len = len(image_names)
        img = Image.open(os.path.join(path, image_names[im_id]))
        img_1 = np.array(img)
        img_1 = smart_resize(img_1, dim)
    else:
        img = crops_1[im_id]
        img_1 = np.array(img)
        iter_len = 2 * len(crops_1)

    for i in range(iter_len):

        x_tr_1.append(img_1)  # True img to compare with

        contr = np.random.randint(0, 15)  # в 1/15 случаев будем применять какие-то изменения к фотке
        crop = np.random.randint(0, 15)
        rotate = np.random.randint(0, 15)

        if contr == 1:
            scale_value = np.random.uniform(0.4, 1.6)
            img = ImageEnhance.Contrast(img).enhance(scale_value)

        if crop == 1:
            img = img.crop((np.random.randint(0, 4), np.random.randint(0, 4),
                            img.size[0] - np.random.randint(0, 4), img.size[1] - np.random.randint(0, 4)))

        if rotate == 1:
            rotate_level = np.random.randint(-5, 5)
            img = img.rotate(rotate_level)
            img.crop((5, 5, img.size[0] - 5, img.size[1] - 5))

        img_2 = np.array(img)
        img_2 = smart_resize(img_2, dim)
        x_tr_2.append(img_2)

        y_tr.append(1)

    for i in range(iter_len - 1):

        x_tr_1.append(img_1)

        if path is not None:
            if i != im_id:
                img_2 = Image.open(os.path.join(path, image_names[i]))  # любая картинка, кроме img_1
        else:
            if i != im_id and i < len(crops_1):
                img_2 = crops_1[i]
                img_2 = np.array(img_2)
                img_2 = smart_resize(img_2, dim)
                x_tr_2.append(img_2)
                y_tr.append(0)

            elif i != im_id and i > len(crops_1):  # added this to double amt of pics (iter_len x2)
                img_2 = crops_1[i - len(crops_1)]
                rotate_level = np.random.randint(-5, 5)
                img_2 = img_2.rotate(rotate_level)
                img_2.crop((5, 5, img_2.size[0] - 5, img_2.size[1] - 5))
                img_2 = np.array(img_2)
                img_2 = smart_resize(img_2, dim)
                x_tr_2.append(img_2)
                y_tr.append(0)

    X = np.array(list(zip(x_tr_1, x_tr_2)))

    x_train, x_test, y_train, y_test = train_test_split(X, y_tr, test_size=0.15, random_state=42)

    x1_train, x2_train = x_train[:, 0], x_train[:, 1]
    x1_test, x2_test = x_test[:, 0], x_test[:, 1]

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x1_train, x2_train, y_train, x1_test, x2_test, y_test


def make_pile_model(x1_train, x2_train, y_train, x1_test, x2_test, y_test, nr_wood_cr1):
    """
    Функция создания модели НС для каждого бревна из df_1.

    :param x1_train:
    :param x2_train:
    :param y_train:
    :param x1_test:
    :param x2_test:
    :param y_test:
    :param nr_wood_cr1:
    :return:
    """
    input_1 = layers.Input(shape=(64, 64, 3), name="img_1")
    input_2 = layers.Input(shape=(64, 64, 3), name="img_2")

    x = layers.Conv2D(32, 3, activation="relu")(input_1)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Conv2D(32, 3, activation='relu', strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Conv2D(64, 3, activation='relu', strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.Conv2D(128, 3, activation='relu', strides=2)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Conv2D(32, 3, activation="relu")(input_2)
    y = layers.Conv2D(32, 3, activation="relu")(y)
    y = layers.Conv2D(64, 3, activation='relu', strides=2)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(64, 3, activation="relu")(y)
    y = layers.Conv2D(64, 3, activation="relu")(y)
    y = layers.Conv2D(64, 3, activation='relu', strides=2)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(128, 3, activation="relu")(y)
    y = layers.Conv2D(128, 3, activation="relu")(y)
    y = layers.Conv2D(128, 3, activation='relu', strides=2)(y)
    y = layers.BatchNormalization()(y)

    con = layers.concatenate([x, y])

    con = layers.Flatten()(con)
    con = layers.Dense(128, activation='relu')(con)
    com = layers.Dropout(0.1)(con)
    output = layers.Dense(1, activation='sigmoid')(con)

    model = tensorflow.keras.Model([input_1, input_2], output, name="my_model")

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit([x1_train, x2_train], y_train, batch_size=4, epochs=5,
                        validation_data=((x1_test, x2_test), y_test))
    counter = 0
    status = 'Success'
    while (history.history['val_accuracy'][-1] < 0.85):
        history = model.fit([x1_train, x2_train], y_train, batch_size=4, epochs=4,
                            validation_data=((x1_test, x2_test), y_test))
        counter += 1
        print('+=+=+=+=+=+=+=+=+=', counter)
        print(history.history['val_accuracy'][-1])
        if history.history['val_accuracy'][-1] > 0.85:
            status = 'Success'
        if counter == 3:
            print('не получилось обучить модель на этапе', nr_wood_cr1)
            if history.history['val_accuracy'][-1] < 0.85:
                status = 'Fail'
            break

    return model, status


def compare_images_geofilter(img_dir_1, img_dir_2, df_1, df_2, dim, acc_margin, default_model_path, neighbours_list):
    """
    Функция сравнения изображений лесовоза с учетом расположения каждого бревна в штабеле. Нейронка здесь будет искать
    похожие бревна на 2й картинке в определенном небольшом радиусе от расположения бревна на 1 картинке. Обе картинки
    "отцентрованы" по координатам штабеля (максимальные и минимальные координаты штабеля). Если штабель во время перевозки
    значительно изменил свою форму, этот метод может плохо отработать. В таком случае лучше использовать compare_images().
    Также эта функция обучает нейронки с нуля для каждого бревна, поэтому обработка может занять много времени.

    :param img_dir_1: директория первого изображения для операции сравнения
    :param img_dir_2: директория второго изображения для операции сравнения
    :param df_1: первый датафрейм (выход функции predict_timber() для 1й картинки)
    :param df_2: второй датафрейм (выход функции predict_timber() для 2й картинки)
    :param default_model_path: директория, в которой хранятся веса модели для сравнения картинок
    :param dim: размеры картинки(каждого бревна) для сравнения
    :param acc_margin: точность, ниже которой выход нейронки будет считаться недействительным (нейронка не нашла
    совпадений по бревну с 1 картинки у бревен со второй картинки)
    :param neighbours_list: список с бревнами - "соседями". Каждому бревну с 1й картинки выбираются близкие по
    координатам бревна со 2й каотинки.

    :return:
    match_dict - словарь, сопоставляющий бревна с первой картинки бревнам со второй (согласно их порядку в df)
    """
    def_model = load_model(default_model_path)

    img_1 = Image.open(img_dir_1)
    img_2 = Image.open(img_dir_2)

    bboxes_1 = get_bbox_from_df(df_1['bbox'].values)
    bboxes_2 = get_bbox_from_df(df_2['bbox'].values)

    crops_1 = []
    crops_2 = []
    match_dict = {}
    stat_res = []

    # создадим resize подборку нарезанных каринок из 2й картинки
    for j in range(len(df_2)):
        x_cntr, y_cntr, w, h = bboxes_to_int(img_2, bboxes_2[j], PIL=True)
        x_min, x_max, y_min, y_max = int(x_cntr - w / 2), int(x_cntr + w / 2), int(y_cntr - h / 2), int(y_cntr + h / 2)

        img_crop_2 = img_2.crop((x_min, y_min, x_max, y_max))

        img_crop_2 = ImageOps.fit(img_crop_2, dim, Image.ANTIALIAS)
        crops_2.append(img_crop_2)
        match = [[] for _ in range(len(df_1))]

    for i in range(len(df_1)):
        x_cntr, y_cntr, w, h = bboxes_to_int(img_1, bboxes_1[i], PIL=True)
        x_min, x_max, y_min, y_max = int(x_cntr - w / 2), int(x_cntr + w / 2), int(y_cntr - h / 2), int(y_cntr + h / 2)

        img_crop_1 = img_1.crop((x_min, y_min, x_max, y_max))
        img_crop_1 = ImageOps.fit(img_crop_1, dim, Image.ANTIALIAS)  # change to smart_resize?
        crops_1.append(img_crop_1)

        img_crop_1 = np.array(img_crop_1)

    for i in range(len(df_1)):
        def_model_res = []

        x1_train, x2_train, y_train, x1_test, x2_test, y_test = get_train_batch(path=None, dim=(64, 64), im_id=i,
                                                                                crops_1=crops_1)
        model, status = make_pile_model(x1_train, x2_train, y_train, x1_test, x2_test, y_test, i)

        stat_res.append(status)

        # если модель успешно обучилась
        if status == 'Success':

            for j in neighbours_list[i]:
                print(i)
                print(neighbours_list[i])
                img_crop_2 = np.array(crops_2[j])

                result = model.predict([np.expand_dims(img_crop_1, 0), np.expand_dims(img_crop_2, 0)])
                print(result)
                print(result.shape)

                match[i].append(result)

        elif status == 'Fail':  # или по диаметру просто?

            for j in neighbours_list[i]:
                img_crop_2 = np.array(crops_2[j])

                result = def_model.predict([np.expand_dims(img_crop_1, 0), np.expand_dims(img_crop_2, 0)])
                def_model_res.extend(result)

            top_3 = [sorted(def_model_res)[-k] for k in range(1, 4)]

            possible_match_idx = [def_model_res.index(top_3[i]) for i in range(len(top_3)) if
                                  top_3[i] > acc_margin]  # берем топ 3 совпадения по выходу нейронки c acc > 0.9)
            print(possible_match_idx)
            possible_areas = find_nearest(df_2['area, dm2'].values, df_1['area, dm2'][i], ret='idx', amt=3)
            if len(possible_match_idx) > 0:
                # проверяем, есть ли среди ближайших значений площадей из df_2 те же индексы,
                # что и при проверке на соответствие картинок
                for m in range(len(possible_match_idx)):
                    if possible_match_idx[m] in possible_areas:
                        idx_win = possible_match_idx[m]
                        break
                    else:
                        idx_win = None
                if idx_win == None:  # ищем ближайшие площади к искомой среди топа выхода нейронки
                    p_areas = np.array([df_2['area, dm2'][idx] for idx in possible_match_idx])
                    idx_win = possible_match_idx[list(p_areas).index(
                        find_nearest(p_areas, df_1['area, dm2'][i], ret='value', amt=1))]  # выбираем 1
            else:
                idx_win = 'нет_совпадений'

            match[i].append(idx_win)

    for i, res in enumerate(match):
        idx_best_res = neighbours_list[i][np.argmax(res)]
        match_dict[i] = idx_best_res

    return match_dict


def check_image(IMG_DIR):
    """
    Функция проверки изображения на наличие 4 канала.
    :param IMG_DIR: директория с изображением
    """

    image = Image.open(IMG_DIR)
    sh = np.array(image).shape
    if sh[2] == 4:
        rgb_image = image.convert('RGB')
        rgb_image.save(IMG_DIR)
        print('Пересохранили картинку в RGB')
    else:
        print('Картинка уже в нужном формате')


def check_save_dir():
    """
    Функция проверки директории для сохранения результатов работы нейронных сетей.
    :return:
    detect_dir - директория, где хранятся все результаты.
    """
    detect_dir = max([os.path.join('/content/forest/yolov5/runs/detect', f_name) for
                      f_name in os.listdir('/content/forest/yolov5/runs/detect')], key=os.path.getctime)

    return detect_dir


def support_arr_generator(element, variance=3):
    lst_with_vars = list(range(-variance, variance))
    coord_variations = [[x, y] for x in lst_with_vars for y in lst_with_vars]
    new_el_list = [[element[0] + i, element[1] + j] for [i, j] in coord_variations]

    return new_el_list


def get_neighbour_list(df_1, df_2, img_dir_1, img_dir_2, rad):
    """
    Функция по поиску бревен - "соседей". Каждому бревну с 1й картинки выбираются близкие по координатам бревна со
    2й каотинки.
    
    :param df_1: df первой картинки (от функции predict_timber())
    :param df_2: df второй картинки (от функции predict_timber())
    :param img_dir_1: ссылка на первую картинку
    :param img_dir_2: ссылка на вторую картинку
    :param rad: коэффициент поиска бревен j картинки 2 в радиусе rad * width и rad * height бревна
    i картинки 1, где width и height - размеры бревна i. Обе картинки приведены к единой системе координат
    на основе предположения, что перевозимый штабель древесины не менял своей геометрической формы и бревна не меняли
    свое расположение внутри штабеля.

    :return: neighbours_list - список бревен картинки 2, которые вероятно распологались рядом с искомым бревном картинки 1.
    """

    bboxes_1 = get_bbox_from_df(df_1['bbox'].values)
    _, _, x_min_1, y_min_1, x_max_1, y_max_1, width_1, height_1 = calc_stack_geometry(bboxes_1, 1, img_dir_1)
    for row in bboxes_1:
        row[0] = row[0] - x_min_1 / width_1
        row[1] = row[1] - y_min_1 / height_1

    bboxes_2 = get_bbox_from_df(df_2['bbox'].values)
    _, _, x_min_2, y_min_2, x_max_2, y_max_2, width_2, height_2 = calc_stack_geometry(bboxes_2, 1, img_dir_2)

    for row in bboxes_2:
        row[0] = row[0] - x_min_2 / width_2
        row[1] = row[1] - y_min_2 / height_2

    neighbours_list = []
    for i in range(len(df_1)):
        temp = []
        xc_1, yc_1, w_1, h_1 = bboxes_1[i]
        for j in range(len(df_2)):
            xc_2, yc_2, w_2, h_2 = bboxes_2[j]
            if (xc_2 * width_2 - rad * w_2 * width_2 < xc_1 * width_1 < xc_2 * width_2 + rad * w_2 * width_2) and \
                    (yc_2 * height_2 - rad * h_2 * height_2 < yc_1 * height_1 < yc_2 * height_2 + rad * h_2 * height_2):
                temp.append(j)
        neighbours_list.append(temp)

    return neighbours_list


def calc_laser(x_min, y_min, x_max, y_max, scale_sq, img_piles_path, save_path, color_search=(255, 255, 255),
               color_paint=(200, 200, 200)):
    img = cv2.imread(img_piles_path)

    # height perspective
    color_cells = {}

    for i, col in enumerate(img):
        for j, cell in enumerate(col):
            if cell[0] == color_search[0] and cell[1] == color_search[1] and cell[2] == color_search[2] \
                    and y_min - 30 < i < y_max + 30 and x_min - 30 < j < x_max + 30: # 30 - margin for error of stack calc
                try:
                    temp_val = color_cells[i]  # arr exists already
                    temp_val.extend([j])
                    color_cells[i] = temp_val
                except KeyError as e:
                    color_cells[i] = [j]

    left_margin_total = []
    right_margin_total = []
    for key, vals in color_cells.items():
        left_margin = [min(vals), key]
        right_margin = [max(vals), key]
        left_margin_total.append(left_margin)
        right_margin_total.append(right_margin)

    # width perspective
    color_cells = {}

    for i, col in enumerate(img):
        for j, cell in enumerate(col):
            if cell[0] == color_search[0] and cell[1] == color_search[1] and cell[2] == color_search[2] \
                    and 110 < i < 270 and 100 < j < 400:  # j - x, i - y
                try:
                    temp_val = color_cells[j]  # arr exists already
                    temp_val.extend([i])
                    color_cells[j] = temp_val
                except KeyError as e:
                    color_cells[j] = [i]

    top_margin_total = []
    bottom_margin_total = []
    for key, vals in color_cells.items():
        top_margin = [key, min(vals)]
        bottom_margin = [key, max(vals)]
        top_margin_total.append(top_margin)
        bottom_margin_total.append(bottom_margin)

    # сортируем массивы для получения корректной замкнутой ломанной линии (left-bottom-right-top)
    top_margin_total = sorted(top_margin_total, key=lambda x: x[0], reverse=True)
    bottom_margin_total = sorted(bottom_margin_total, key=lambda x: x[0])
    right_margin_total = list(reversed(right_margin_total))

    # очистка от дупликатов по проекциям
    comb_l_r = []
    comb_l_r.extend(left_margin_total)
    comb_l_r.extend(right_margin_total)
    top_margin_total = [e for e in top_margin_total if e not in comb_l_r]
    bottom_margin_total = [e for e in bottom_margin_total if e not in comb_l_r]

    # доп очистка для bottom и top сегментов (+- 10 пикселей)
    arr_to_del = []
    for el in bottom_margin_total:
        arr = support_arr_generator(el, variance=10)
        for val in arr:
            if val in left_margin_total or val in right_margin_total:
                arr_to_del.append(el)
                break

    bottom_margin_total = [e for e in bottom_margin_total if e not in arr_to_del]

    arr_to_del = []
    for el in top_margin_total:
        arr = support_arr_generator(el, variance=10)
        for val in arr:
            if val in left_margin_total or val in right_margin_total:
                arr_to_del.append(el)
                break

    top_margin_total = [e for e in top_margin_total if e not in arr_to_del]

    # создаем замкнутый контур
    contour = np.concatenate([left_margin_total, bottom_margin_total, right_margin_total, top_margin_total])

    cv2.drawContours(img, np.array(contour).reshape((-1, 1, 2)).astype(np.int32), -1,
                     (color_paint[0], color_paint[1], color_paint[2]), 3)

    cv2.imwrite(os.path.join(save_path, 'contour.jpg'), img)

    surface = Polygon(contour)
    area_pix = surface.area # pixels
    area_dm = area_pix * scale_sq # pix * дм2/pix = дм2

    plt.plot(*surface.exterior.xy)

    return area_dm, img

