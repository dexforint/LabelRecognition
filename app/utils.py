MODEL_INDEX = 1
MODELS_INFO = [
    ("vit_h", "/kaggle/working/Hi-SAM/pretrained_checkpoint/sam_tss_h_hiertext.pth"),
    ("vit_l", "/kaggle/working/Hi-SAM/pretrained_checkpoint/sam_tss_l_hiertext.pth"),
]
MODEL_INFO = MODELS_INFO[MODEL_INDEX]

import numpy as np
import scipy.optimize as opt


# Функция для вычисления расстояний от всех точек до окружности
def calc_R(xc, yc, points):
    """Вычисляем расстояние от каждой точки до центра окружности (xc, yc)."""
    return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)


def f_2(c, points):
    """Функция ошибки: разница между расчетными радиусами и их средним значением."""
    Ri = calc_R(*c, points)
    return Ri - Ri.mean()


def fit_circle(points):
    """Фитинг окружности методом наименьших квадратов."""
    # Начальные приближения: центр в среднем по координатам точек
    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    center_estimate = x_m, y_m
    center, ier = opt.leastsq(f_2, center_estimate, args=(points,))

    # Радиус как среднее расстояние всех точек до центра
    Ri = calc_R(*center, points)
    R = Ri.mean()

    return center, R


def calculate_circularity(points):
    """Оценка того, насколько точки лежат на одной окружности."""
    center, R = fit_circle(points)

    # Вычисляем отклонение расстояний от радиуса
    distances = calc_R(*center, points)
    deviations = np.abs(distances - R)

    # Для оценки можно использовать относительное стандартное отклонение
    relative_std = np.std(deviations) / R

    # Чем меньше относительное стандартное отклонение, тем ближе точки к окружности
    return relative_std


import matplotlib.patches as patches
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Levenshtein import ratio as lev_ratio

import easyocr

ru_reader = easyocr.Reader(["ru"], gpu=True)
# reader = easyocr.Reader(['ru', 'en'], gpu = True)


def visualize_ocr_results(image, ocr_results):
    # Создаем фигуру и оси для визуализации
    fig, ax = plt.subplots(1, figsize=(12, 12))

    # Отображаем изображение на фоне
    ax.imshow(image)

    # Проходим по каждому результату OCR
    for bbox, text, confidence, box in ocr_results:
        # bbox - это список координат углов (верхний левый, верхний правый, нижний правый, нижний левый)
        # Извлекаем координаты верхнего левого угла, ширину и высоту
        top_left = bbox[0]
        top_right = bbox[1]
        bottom_left = bbox[3]

        # Вычисляем ширину и высоту прямоугольника
        width = top_right[0] - top_left[0]
        height = bottom_left[1] - top_left[1]

        # Создаем прямоугольник для bbox
        rect = patches.Rectangle(
            (top_left[0], top_left[1]),
            width,
            height,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )

        # Добавляем прямоугольник на изображение
        ax.add_patch(rect)

        # Добавляем текст рядом с bbox
        ax.text(
            top_left[0],
            top_left[1] - 10,
            f"{text} ({confidence:.2f})",
            color="blue",
            fontsize=12,
        )  # , backgroundcolor='white'

    # Отключаем осевые линии
    # ax.axis('off')

    # Показываем результат
    plt.show()


# if not os.path.exists("./Hi-SAM"):
#     !git clone https://github.com/ymy-k/Hi-SAM.git > /dev/null
#     %cd /kaggle/working/Hi-SAM
#     !pip install -r requirements.txt > /dev/null

#     if MODEL_INDEX == 0:
#         !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./pretrained_checkpoint
#         !wget -q https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_h_maskdecoder.pth -O ./pretrained_checkpoint/vit_h_maskdecoder.pth
#     elif MODEL_INDEX == 1:
#         !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P ./pretrained_checkpoint
#         !wget -q https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_h_maskdecoder.pth -O ./pretrained_checkpoint/vit_l_maskdecoder.pth
# else:
#     %cd /kaggle/working/Hi-SAM


from huggingface_hub import hf_hub_download

MODEL_INFO
hf_hub_download(
    repo_id="dexforint/Hi-SAM-weights",
    filename=MODEL_INFO[1].split("/")[-1],  # "sam_tss_h_hiertext.pth",
    repo_type="dataset",
    local_dir="./pretrained_checkpoint",
)


import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.predictor import SamPredictor
import glob
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon
import pyclipper
import warnings

warnings.filterwarnings("ignore")


def patchify(image: np.array, patch_size: int = 256):
    h, w = image.shape[:2]
    patch_list = []
    h_num, w_num = h // patch_size, w // patch_size
    h_remain, w_remain = h % patch_size, w % patch_size
    row, col = h_num + int(h_remain > 0), w_num + int(w_remain > 0)
    h_slices = [[r * patch_size, (r + 1) * patch_size] for r in range(h_num)]
    if h_remain:
        h_slices = h_slices + [[h - h_remain, h]]
    h_slices = np.tile(h_slices, (1, col)).reshape(-1, 2).tolist()
    w_slices = [[i * patch_size, (i + 1) * patch_size] for i in range(w_num)]
    if w_remain:
        w_slices = w_slices + [[w - w_remain, w]]
    w_slices = w_slices * row
    assert len(w_slices) == len(h_slices)
    for idx in range(0, len(w_slices)):
        # from left to right, then from top to bottom
        patch_list.append(
            image[
                h_slices[idx][0] : h_slices[idx][1],
                w_slices[idx][0] : w_slices[idx][1],
                :,
            ]
        )
    return patch_list, row, col


def unpatchify(patches, row, col):
    # return np.array
    whole = [
        np.concatenate(patches[r * col : (r + 1) * col], axis=1) for r in range(row)
    ]
    whole = np.concatenate(whole, axis=0)
    return whole


def patchify_sliding(image: np.array, patch_size: int = 512, stride: int = 256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j + patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i + patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            h_slice_list.append(h_slice)
            w_slice = slice(start_w, end_w)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])

    return patch_list, h_slice_list, w_slice_list


def unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    assert len(ori_size) == 2  # (h, w)
    whole_logits = np.zeros(ori_size)
    assert len(patch_list) == len(h_slice_list)
    assert len(h_slice_list) == len(w_slice_list)
    for idx in range(len(patch_list)):
        h_slice = h_slice_list[idx]
        w_slice = w_slice_list[idx]
        whole_logits[h_slice, w_slice] += patch_list[idx]

    return whole_logits


def show_points(coords, ax, marker_size=200):
    ax.scatter(
        coords[0],
        coords[1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=0.25,
    )


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = (
            color
            if color is not None
            else np.array([30 / 255, 144 / 255, 255 / 255, 0.5])
        )
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_res(masks, scores, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis("off")
        plt.savefig(filename, bbox_inches="tight", pad_inches=-0.1)
        plt.close()


def show_hi_masks(masks, word_masks, input_points, filename, image, scores):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, (line_para_masks, word_mask, hi_score, point) in enumerate(
        zip(masks, word_masks, scores, input_points)
    ):
        line_mask = line_para_masks[0]
        para_mask = line_para_masks[1]
        show_mask(
            para_mask, plt.gca(), color=np.array([255 / 255, 144 / 255, 30 / 255, 0.5])
        )
        show_mask(line_mask, plt.gca())
        word_mask = word_mask[0].astype(np.uint8)
        contours, _ = cv2.findContours(
            word_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        select_word = None
        for cont in contours:
            epsilon = 0.002 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            pts = unclip(points)
            if len(pts) != 1:
                continue
            pts = pts[0].astype(np.int32)
            if cv2.pointPolygonTest(pts, (int(point[0]), int(point[1])), False) >= 0:
                select_word = pts
                break
        if select_word is not None:
            word_mask = cv2.fillPoly(np.zeros(word_mask.shape), [select_word], 1)
            show_mask(
                word_mask,
                plt.gca(),
                color=np.array([30 / 255, 255 / 255, 144 / 255, 0.5]),
            )
        show_points(point, plt.gca())
        print(f"point {i}: line {hi_score[1]}, para {hi_score[2]}")

    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_binary_mask(mask: np.array, filename):
    if len(mask.shape) == 3:
        assert mask.shape[0] == 1
        mask = mask[0].astype(np.uint8) * 255
    elif len(mask.shape) == 2:
        mask = mask.astype(np.uint8) * 255
    else:
        raise NotImplementedError
    mask = Image.fromarray(mask)
    mask.save(filename)


def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


class Arguments:
    model_type = MODEL_INFO[0]
    device = "cuda"
    checkpoint = MODEL_INFO[1]
    input_size = [1024, 1024]
    patch_mode = False
    attn_layers = 1
    prompt_len = 12
    hier_det = False


args = Arguments()
hisam = model_registry[MODEL_INFO[0]](args)
hisam.eval()
hisam.to("cuda")
predictor = SamPredictor(hisam)


def hisam_mask(image):
    if args.patch_mode:
        ori_size = image.shape[:2]
        patch_list, h_slice_list, w_slice_list = patchify_sliding(
            image, 512, 384
        )  # sliding window config
        mask_512 = []
        for patch in tqdm(patch_list):
            predictor.set_image(patch)
            m, hr_m, score, hr_score = predictor.predict(
                multimask_output=False, return_logits=True
            )
            assert hr_m.shape[0] == 1  # high-res mask
            mask_512.append(hr_m[0])
        mask_512 = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
        assert mask_512.shape[-2:] == ori_size
        mask = mask_512
        mask = mask > predictor.model.mask_threshold
        return (mask[0] * 255).astype(np.uint8)
    else:
        predictor.set_image(image)
        if args.hier_det:
            input_point = np.array([[125, 275]])  # for demo/img293.jpg
            input_label = np.ones(input_point.shape[0])
            mask, hr_mask, score, hr_score, hi_mask, hi_iou, word_mask = (
                predictor.predict(
                    multimask_output=False,
                    hier_det=True,
                    point_coords=input_point,
                    point_labels=input_label,
                )
            )
            show_hi_masks(hi_mask, word_mask, input_point, out_filename, image, hi_iou)
        else:
            mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
            return (hr_mask[0] * 255).astype(np.uint8)


db = pd.read_excel(
    "/kaggle/input/marking24-dataset/data/ДеталиПоПлануДляРазрешенныхЗаказов.xlsx"
)


# Заполняем базу данных
db_data = []
article_nums = set()
lcounter = {}
allowlist = set()
for index in range(len(db)):
    row = db.iloc[index]

    article_num = row["ДетальАртикул"].strip()

    if article_num.startswith('"'):
        article_num = article_num[1:]

    if article_num.endswith('"'):
        article_num = article_num[:-1]

    article_num = article_num.strip()

    splits = article_num.split()

    if len(splits) == 1:
        article_num = splits[0]
    else:
        article_num = " ".join(splits[:-1])

    article_nums.add(article_num)

    if not np.isnan(row["ПорядковыйНомер"]):
        serial_number = float(row["ПорядковыйНомер"])
        serial_number = int(serial_number)
        serial_number = str(serial_number)
        serial_number = serial_number.replace(".", "")  # !!!!!!!!!!!!!
        text = article_num + " " + str(serial_number)
    else:
        serial_number = None
        text = article_num

    for s in text:
        allowlist.add(s.lower())
        allowlist.add(s.upper())

        lcounter[s] = lcounter.get(s, 0) + 1

    db_data.append(
        {
            "db_index": index,
            "article_num": article_num,
            "serial_number": serial_number,
            "text": text.lower(),
            "orig_text": text,
        }
    )

db_data.append(
    {
        "db_index": None,
        "article_num": "АНЕМ.492664.500-03LP",
        "serial_number": "497/2",
        "text": "АНЕМ.492664.500-03LP 497/2".lower(),
        "orig_text": "АНЕМ.492664.500-03LP 497/2",
    }
)


# allowlist = "".join(list(allowlist))
# allowlist


def find_similar_in_db(text, in_=None, last_num=None):
    """Функция для поиска похожего текста в базе данных"""
    global db_data

    if last_num:
        last_num = f" {last_num}"

    text = text.lower()

    sims = []
    for sample in db_data:
        sample_text = sample["text"]
        if last_num:
            if not sample_text.endswith(last_num):
                continue

        if in_:
            for in_el in in_:
                if not (in_el in sample_text):
                    continue

        val = lev_ratio(text, sample_text)
        sims.append((sample["orig_text"], val, sample["db_index"]))

    sims.sort(key=lambda el: -el[1])
    return sims[0]


find_similar_in_db("195-30-1286 3089")


def get_detail_info(index):
    row = db.iloc[index]
    article_num = row["ДетальАртикул"]
    num = row["ПорядковыйНомер"]
    if np.isnan(num):
        num = "None"
    title = row["ДетальНаименование"]
    order = row["ЗаказНомер"]
    station = row["СтанцияБлок"]
    return f"Информация об изделии:\nАртикул: {article_num}\nНомер: {num}\nНаименование: {title}\nЗаказ: {order}\nСтанция: {station}"


def visualize(sample):
    if "img" in sample:
        img = sample["img"]
    else:
        img_path = sample["img_path"]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W, _ = img.shape

    print(sample["label_text"])
    sims = find_similar_in_db(sample["label_text"])

    for box in sample["boxes"]:
        x, y, w, h = box
        x = int(x * W)
        y = int(y * H)
        w = int(w * W)
        h = int(h * H)

        x1 = x - w // 2
        x2 = x + w // 2
        y1 = y - h // 2
        y2 = y + h // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def get_box(points):
    """Функция для перевода набора точек в bbox"""
    x1 = min(p[0] for p in points)
    x2 = max(p[0] for p in points)
    y1 = min(p[1] for p in points)
    y2 = max(p[1] for p in points)

    return (x1, y1, x2, y2)


def ocr(image, is_polar=False):
    """Функция для распознавания текста на изображении"""
    bounds = ru_reader.readtext(
        image,
        # decoder="greedy", # greedy beamsearch wordbeamsearch
        batch_size=10,
        workers=4,
        allowlist=" -0123456789" if is_polar else None,
        # 'вВnИлoi[ОаMМO7ос2Aб5ПЭaСм.-4гр83ЧтнLZузpч91]XxЕФиТеУIz/NРГэпЗk0ЛP Аl6кфКНKmБ'
        blocklist="ЪъШш{}()Жж'|`:;",
        # min_size=10,
    )

    bounds = [(points, text, prob, get_box(points)) for (points, text, prob) in bounds]
    return bounds


non_zero_indices = None


def warpPolar(mask):
    """Функция для получения полярного искажения маски"""
    global non_zero_indices
    h, w = mask.shape[0], mask.shape[1]

    non_zero_indices = np.where(mask > 0)

    # x1, y1 - самые левые и верхние ненулевые значения
    y1 = np.min(non_zero_indices[0])  # минимальный индекс по оси y
    x1 = np.min(non_zero_indices[1])  # минимальный индекс по оси x

    # x2, y2 - самые правые и нижние ненулевые значения
    y2 = np.max(non_zero_indices[0])  # максимальный индекс по оси y
    x2 = np.max(non_zero_indices[1])  # максимальный индекс по оси x

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    print("center:", center)
    radius = (x2 - x1) / 2

    # y, x = np.nonzero(mask)
    # x_center = np.mean(x)
    # y_center = np.mean(y)
    # center = (x_center, y_center)
    # print(center)
    polar_image = cv2.warpPolar(mask, (w, h), center, w // 2, cv2.WARP_POLAR_LINEAR)

    polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    polar_image = np.concatenate([polar_image, polar_image], axis=1)

    return polar_image


# polar_image = warpPolar(mask)
# plt.imshow(polar_image)
# plt.show()


import re


def sort_bounds(bounds, direction="down"):
    """Сортирум предсказания (боксы) в зависимости от желаемого направления"""
    if direction == "down":
        bounds.sort(key=lambda bound: bound[3][1])
    elif direction == "right":
        bounds.sort(key=lambda bound: bound[3][0])
    else:
        assert False
    return bounds


def remove_boundary_bounds(bounds, width):
    """Удаляем граничные предсказания (для полярного искажения)"""
    x1 = bounds[0][3][0]

    if x1 < 20:
        bounds = bounds[1:]

    x2 = bounds[-1][3][2]

    if (width - x2) < 20:
        bounds = bounds[:-1]

    return bounds


def filter(bounds, width, is_polar=False):
    """Функция для фильтрации ненужных предсказаний"""
    global bad_nums
    # фильтруем bounds по размеру и (возможно) по prob и по расположению - удаляем дубликаты
    # удаляем 54
    if is_polar:
        bounds = remove_boundary_bounds(bounds, width)

    filtered_bounds = []
    lengths = []
    for points, text, prob, box in bounds:
        text = text.strip()
        if is_polar and (text == "54" or len(text) < 3):
            continue

        if len(text) < 4 and "54" in text:
            continue

        if prob < 0.35:
            continue

        lengths.append((box[2] - box[0], box[1]))
        filtered_bounds.append((points, text, prob, box))

    if not is_polar:
        return filtered_bounds

    if len(lengths) == 0:
        return []

    bounds = []
    # !Изменить !!!!!!!! Находим по разрыву
    max_length, target_h = max(lengths, key=lambda el: el[0])
    flag = False
    for points, text, prob, box in filtered_bounds:
        if flag:
            h = box[1]
            if h == target_h:
                break

            bounds.append((points, text, prob, box))
        else:
            length = box[2] - box[0]
            if length == max_length:
                flag = True
                bounds.append((points, text, prob, box))

    return bounds


def union(bounds):
    """Объединяем предсказания, если они находятся близко"""
    pass


def remove_alpha_words(text):
    """Функция для удаления целобуквенных слов из текста"""
    # Регулярное выражение для нахождения слов, состоящих только из букв
    pattern = r"\b[a-zA-Zа-яА-ЯёЁ]{5,}\b"
    # Замена таких слов на пустую строку
    text = re.sub(pattern, "", text)
    # Удаление лишних пробелов
    text = re.sub(r" +", " ", text)
    return text


def get_text(bounds):
    """Получаем текст из результата распознавания текста"""
    if len(bounds) == 0:
        return None, []
    # print("Mean Prob:", sum([el[2] for el in bounds]) / len(bounds))

    in_ = []
    for bound in bounds:
        word = bound[1]
        prob = bound[2]
        if prob > 0.95:
            in_.append(word)

    text = " ".join([el[1] for el in bounds])
    text = remove_alpha_words(text)
    return text, in_


def spec_case1(bounds):
    # много линий, в самом низу - число
    pass


def get_text_direction(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sum_area = 0.0
    elongation = 0.0
    angles = []
    points = []
    for cnt in contours:
        minAreaRect = cv2.minAreaRect(cnt)
        points.append(minAreaRect[0])
        box = cv2.boxPoints(minAreaRect)

        box = np.int0(box)  # Округляем координаты до целых чисел

        boundingRect = cv2.boundingRect(cnt)

        x, y, w, h = boundingRect
        area = w * h

        if w > h * 1.2:
            elongation += area
        else:
            elongation -= area

        sum_area += area

        angle = minAreaRect[-1]
        if angle > 45:
            angle = 90 - angle
        angles.append(angle)

    angle_std = np.std(angles)
    print("angle_std:", angle_std)

    points = np.array(points)
    circularity = calculate_circularity(points)
    print("circularity:", circularity)

    elongation = elongation / sum_area
    print("elongation:", elongation)

    if elongation > 0.9:
        return "side"

    if elongation < -0.9:
        return None

    if circularity < 0.08 and angle_std > 10.0:  #  and angle_std > 15.0
        return "circle"

    return None


import time
from tqdm.auto import tqdm

bounds = None
polar_image = None
global_candidates = []


def func(mask, rotate=0, candidates=None):
    """Функция для получения текста путём поворота изображения"""
    global bounds, global_candidates
    width = mask.shape[1]

    if rotate == 1:
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 2:
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    bounds = ocr(mask)
    bounds = sort_bounds(bounds, direction="down")
    bounds = filter(bounds, width)
    if len(bounds) > 1:
        if bounds[0][1].lower().startswith("ам"):
            return get_text([bounds[0], bounds[-1]])
    text, in_ = get_text(bounds)
    return text, in_


def polar_func(mask, candidates=None):
    """Функция для получения текста путём полярного искажения изображения"""
    global global_candidates, bounds, polar_image
    width = mask.shape[1]

    polar_image = warpPolar(mask)
    bounds = ocr(polar_image, is_polar=True)
    # print(bounds)
    bounds = sort_bounds(bounds, direction="right")
    bounds = filter(bounds, width, is_polar=True)
    text, in_ = get_text(bounds)
    return text, in_


def get_bbox(mask):
    non_zero_indices = np.where(mask > 0)

    # x1, y1 - самые левые и верхние ненулевые значения
    y1 = np.min(non_zero_indices[0])  # минимальный индекс по оси y
    x1 = np.min(non_zero_indices[1])  # минимальный индекс по оси x

    # x2, y2 - самые правые и нижние ненулевые значения
    y2 = np.max(non_zero_indices[0])  # максимальный индекс по оси y
    x2 = np.max(non_zero_indices[1])  # максимальный индекс по оси x

    return x1, y1, x2, y2


def error_correction(text):
    global error_map
    for error, correction in error_map.items():
        text = text.replace(error, correction)

    return text


def pipeline(image):
    """Полный пайплан работы"""
    global bounds, polar_image

    t1 = time.time()
    candidates = []

    # получаем маску текста
    mask = hisam_mask(image)
    bbox = get_bbox(mask)
    # mask = 255 - mask

    text_direction = get_text_direction(mask)
    print("text direction:", text_direction)

    if text_direction == "circle":
        text, in_ = polar_func(mask)
        if text:
            candidates.append(find_similar_in_db(text, in_=in_))
    elif text_direction == "side":
        text, in_ = func(mask, rotate=1)
        if text:
            candidates.append(find_similar_in_db(text, in_=in_))

        text, in_ = func(mask, rotate=2)
        if text:
            candidates.append(find_similar_in_db(text, in_=in_))
    # else:
    text, in_ = func(mask, rotate=0)
    if text:
        candidates.append(find_similar_in_db(text, in_=in_))

    best_candidate = max(candidates, key=lambda el: el[1])
    t2 = time.time()
    # print("Time:", round(t2 - t1, 2))
    return best_candidate, bbox
    # find_similar_in_db


def recognize(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    best_candidate, bbox = pipeline(img)
    detail_info = get_detail_info(best_candidate[-1])
    return detail_info


def bbox2label(bbox, W, H):
    x1, y1, x2, y2 = bbox
    x = (x2 + x1) / 2 / W
    y = (y2 + y1) / 2 / H

    w = (x2 - x1) / W
    h = (y2 - y1) / H

    return f"0 {x} {y} {w} {h}\n"


except_paths = []


def full_pipeline(images_dir):
    global except_paths
    image_files = list(os.listdir(images_dir))
    image_files.sort()

    table = []

    for image_file in tqdm(image_files):
        if not image_file.endswith((".png", ".jpg", ".jpeg", ".JPG")):
            continue

        image_path = os.path.join(images_dir, image_file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            (article_num, score, db_index), bbox = pipeline(img)
        except:
            except_paths.append(image_path)
            print("Exception:", image_path)

        label = bbox2label(bbox, img.shape[1], img.shape[0])

        table.append((image_file, label, f'"{article_num}"'))

    table = pd.DataFrame(table, columns=["image_file", "label", "label_text"])
    table.to_csv("/kaggle/working/test_submission2.csv", index=False)
    return table
