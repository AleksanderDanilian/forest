import math

import cv2
import numpy as np
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
        radius = int(sq ** 0.5 / 2)  # sq?
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


def draw_classes(img, bboxes, w_class_list, detect_dir, color=(255, 0, 0), text_color=(255, 255, 0), thickness=2):
    """
    Функция подсчета общей площади бревен
    на вход: изображение, bbox
    на выход: изображение с отметкой бревна и площадь бревна
    """

    height, width = img.shape[0], img.shape[1]
    for i in range(len(bboxes)):
        x_cntr, y_cntr, w, h = bboxes[i]
        x_cntr, w = int(x_cntr * width), int(w * width)
        y_cntr, h = int(y_cntr * height), int(h * height)

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


def find_nearest(a, a0):
    """
    Поиск элемента в массиве `a` ближайшего к скаляру `a0`
    """
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]
