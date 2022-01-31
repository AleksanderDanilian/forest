import os
import cv2
import pandas as pd
import PIL
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
from helper_functions import plate_detector, visualize, prepare_crops, calc_stack_geometry, get_GPS, draw_classes
from tensorflow.keras.models import load_model


def predict_timber(w_length, weights_yolov5, weights_class, img_dir, path,
                   conf=0.7, bbox_type='ellipse', final_wide=800):
    """ Основная функция. Находит и рассчитывает площадь номера.
    Предсказывает координаты и размер бревен, рассчитывает площадь бревен.
    на вход:
    img_dir - путь к изображению
    w_length - длина древесины
    weighns_yolov5 - обученные веса yolo
    weighns_class - обученные веса нейронки по классификации сортов древесины
    path - куда сохраняем результаты
    conf - уверенность
    bbox_type - тип bounding box, варианты: ellipse, circle, box
    final_wide - ширина выводимого изображения
    show - визуализировать результат (boolean)

    на выходе:
    img_edited - изображение шириной final_wide c нанесенными кругами
    df - датафрейм со всеми данными по древесине
    w_volume - обем древесины в кузове
    text_arr - распознанный номер
    """

    # запуск предсказание yolo
    path_to_detect_py = '/content/forest/yolov5/detect.py'
    os.system(
        f"python {path_to_detect_py} --weights {weights_yolov5} --img 640 --save-txt --conf {conf} --source {img_dir}")

    # определение папки с последним предсказанием yolo в папке /content/yolov5/runs/detect
    detect_dir = max([os.path.join(path, dir) for dir in os.listdir(path)], key=os.path.getctime)

    # чтение предсказанных координат
    bbox_array = np.loadtxt(detect_dir + '/labels/' + os.listdir(detect_dir + '/labels/')[0])

    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = bbox_array[:, 1:]

    # функция распознавания номера
    plate, text_arr = plate_detector(image)
    pi = 3.14159
    # меняем изображение до подготовленных размеров
    r = float(final_wide) / image.shape[1]
    dim = (final_wide, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    if len(plate):
        plate[:, :, 0] = plate[:, :, 0] * resized.shape[0]
        plate[:, :, 1] = plate[:, :, 1] * resized.shape[1]

        pts = np.array(plate, np.int32)

        resized = cv2.polylines(resized, pts, True, (0, 0, 255), thickness=2)

    if len(plate):
        area = Polygon(plate[0]).area  # площадь номера в пикселях
    else:
        area = 4000  # средний размер номера

    img, areas_list, scale_sq = visualize(resized, bboxes, area=area, bbox_type=bbox_type)

    model = load_model(weights_class)
    w_class_list = []
    w_classes = ['1', '3', 'dr']  # подвиды древесины (для модели EfficientNetB0)
    for i in range(len(bboxes)):
        cropped_img = prepare_crops(img_dir, bboxes[i], os.path.join(detect_dir, 'crops'), f'{i}.png',
                                    resize_dim=(64, 64), res_type='zoom', normalize=False)
        w_class = w_classes[np.argmax(model.predict(np.expand_dims(cropped_img, axis=0)))]
        w_class_list.append(w_class)

    d_list = list(map(lambda x: round((x / pi) ** 0.5 * 20, 2), areas_list))  # диаметры бревен (2*10)
    s_overall = round(sum(areas_list) / 100, 2)  # дм2 в м2
    areas_list = list(map(lambda x: round(x, 2), areas_list))

    draw_classes(resized, bboxes, w_class_list, detect_dir)  # here - from resized - draw wood classes

    w_volume = round(w_length / 100 * s_overall, 2)  # пользователь вводит см
    stack_width, stack_height = calc_stack_geometry(bboxes, scale_sq, img_dir)

    img_edited = Image.fromarray(img, 'RGB')
    img_edited.save(detect_dir + f'/{s_overall}_{w_volume}_{text_arr}.png')

    bb_pandas = [str(i) for i in bboxes]  # по другому список не положить в 1 колонку в пандас

    data = np.vstack((d_list, areas_list, w_class_list, bb_pandas)).transpose()

    df = pd.DataFrame(data=data,
                      columns=['diameter, cm', 'area, dm2', 'wood class', 'bbox'])
    df = df.astype({'diameter, cm': 'float64', 'area, dm2': 'float64'})

    df.to_csv(detect_dir + f'/{s_overall}_{w_volume}_{text_arr}.csv')

    coords_gps = get_GPS(img_dir)  # извлекаем гео метки

    return df, img_edited, w_volume, text_arr, stack_width, stack_height, coords_gps
