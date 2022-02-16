import math
import re

import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path

import PIL
from PIL import Image, ImageOps

import random
import os
import subprocess
import sys

import timeit
import time

from shapely.geometry import Polygon

from GPSPhoto import gpsphoto

# Функции визуализации 

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (40, 0, 0)  # White
pi = 3.14159
PLATE_AREA = 5.2 * 1.1


def visualize_bbox(img, bbox, area=0, color=BOX_COLOR, thickness=2, bbox_type='ellipse'):
    """
    Функция подсчета общей площади бревен
    на вход: изображение, bbox
    на выход: изображение с отметкой бревна и площадь бревна
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
    на вход: изображение
    на выход: координаты углов и результат распознавания номера
    """
    img = image.copy()
    bbxs = bboxes.copy()
    area_list = []  # список площадей каждого бревна

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
    на вход: изображение
    на выход: координаты углов и результат распознавания номера
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
    """
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def thumbnail(pil_img, dims):
    """
    Функция для обрезки изображения по наибольшему размеру.
    На вход: изображение
    На выход: квадратное изображение размером (dims, dims) без искажения исходного отношения высоты и ширины.
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
    функция подготавливает вырезанные bbox для
    подачи в модель по классификации изображений
    и сохраняет нарезки изображений в папку
    на вход: стартовое изображение, bbox
    на выход: numpy array, готовый для подачи в модель
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
    с древесиной
    на вход: все bboxes, размеры номера
    на выход: координаты рамки, высота, ширина
    """
    image = Image.open(img_dir)
    width, height = image.size

    xMin, yMin, xMax, yMax = width, height, 0, 0
    for i in range(len(bboxes)):
        xc, yc, w, h = bboxes[i]
        crop_box = [int(xc * width - w * width / 2), int(yc * height - h * height / 2), int(xc * width + w * width / 2),
                    int(yc * height + h * height / 2)]
        if crop_box[0] < xMin:  # левый верхний угол
            xMin = crop_box[0]
        if crop_box[1] < yMin:  # левый верхний угол
            yMin = crop_box[1]
        if crop_box[2] > xMax:  # правый нижний угол
            xMax = crop_box[2]
        if crop_box[3] > yMax:  # правый нижний угол
            yMax = crop_box[3]

    scale = 10 * math.sqrt(scale_sq)  # линейный масштаб cм/pixel (был дм2/pixel2)

    stack_height = (yMax - yMin) * scale
    stack_width = (xMax - xMin) * scale

    return stack_width, stack_height


def get_GPS(img_dir):
    """
    Функция извлекает геоданные из фотографии, если они там есть
    """
    data = gpsphoto.getGPSData(img_dir)
    if data != {}:
        gps_coords = {'Широта': data['Latitude'], 'Долгота': data['Longitude']}
    else:
        gps_coords = 'У фотографии нет гео метки'
    return gps_coords


def bboxes_to_int(img, bboxes, PIL=False):
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
    Функция подсчета общей площади бревен
    на вход: изображение, bbox
    на выход: изображение с отметкой бревна и площадь бревна
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
    Поиск элемента в массиве `arr` ближайшего к скаляру `value`
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
    param df_1: Датафрейм первой картинки
    param df_2: Датафрейм второй картинки
    param margin: диапазон поиска бревен на 1й картинке в процентах от значения площади конкретного бревна на
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
    x_cntr, y_cntr, w, h = bboxes_to_int(img, bboxes, PIL=False)

    # x_min, x_max, y_min, y_max = int(x_cntr - w / 2), int(x_cntr + w / 2), int(y_cntr - h / 2), int(y_cntr + h / 2)
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
    Функция рисования номеров бревен
    на вход: изображения, датафреймы с данными, спи
    на выход: изображение с отметкой бревна и площадь бревна
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
    param df_bbox_values: df['bbox'].values
    return: list of bbox coordinates
    """
    bboxes = []

    for el in df_bbox_values:
        el = el.strip('[]').strip()
        el = re.findall(r'0\.\d{0,5}', el)
        bboxes.append(list(map(float, el)))

    return bboxes


def compare_images(img_dir_1, img_dir_2, df_1, df_2, model_path, dim, acc_margin):
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


def check_image(IMG_DIR):

  image = Image.open(IMG_DIR)
  sh = np.array(image).shape
  if sh[2] == 4:
    rgb_image = image.convert('RGB')
    rgb_image.save(IMG_DIR)
    print('Пересохранили картинку в RGB')
  else:
    print('Картинка уже в нужном формате')