"""
元ソース：Port of Single Shot MultiBox Detector to Keras
https://github.com/rykov8/ssd_keras
SSD_training.ipynb
"""

import sys
sys.path.append(r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\AI_Edge_Contest\object_detection\SSD_classes_py\all_SSD_module\SSD')

import random
import os
import numpy as np
import PIL.ImageFilter
from random import shuffle
from scipy.misc import imresize
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import transform_matrix_offset_center
#from keras.preprocessing.image import apply_affine_transform
from keras.preprocessing.image import apply_transform
from dtc_util_edit import load_img
from ssd_vgg import input_shape
from ssd_vgg import preprocess_input


class Generator(object):
    def __init__(self, boxes, bbox_util,
                 train_path_list, val_path_list, image_size, batch_size):
        self.boxes = boxes
        self.bbox_util = bbox_util
        self.train_file_paths = train_path_list
        self.val_file_paths = val_path_list
        self.train_batches = len(train_path_list)
        self.val_batches = len(val_path_list)
        self.image_size = image_size
        self.batch_size = batch_size

        self.min_box_size = 0.2
        self.saturation_var = 0.50
        self.brightness_var = 0.50
        self.contrast_var = 0.50
        self.lighting_std = 0.50
        self.blur_prob = 0.0
        self.blur_intensity = 1
        self.hflip_prob = 0.5
        self.vflip_prob = 0.0
        self.zoom_prob = 0.3
        self.zoom_range = (0.9, 1.1)
        self.shift_prob = 0.5
        self.shift_wrg = 0.05
        self.shift_hrg = 0.05
        self.crop_prob = 0.9
        self.crop_area_range = [0.7, 1.0]
        self.crop_aspect_range = [4.0/5.0, 5.0/4.0]

        self.color_jitter = []
        if self.saturation_var:
            self.color_jitter.append(self.saturation)
        if self.brightness_var:
            self.color_jitter.append(self.brightness)
        if self.contrast_var:
            self.color_jitter.append(self.contrast)

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def blur(self, img, radius=1):
        if np.random.random() < self.blur_prob:
            img = img.filter(PIL.ImageFilter.GaussianBlur(radius=radius))
        return img

    def horizontal_flip(self, img, boxes):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        return img, boxes

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1, :]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_zoom(self, img, boxes):
        if self.zoom_prob < np.random.random():
            return img, boxes

        original_img = np.copy(img)
        original_boxes = np.copy(boxes)

        r = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_matrix = np.array([[r, 0, 0],
                                [0, r, 0],
                                [0, 0, 1]])

        img_h = img.shape[0]
        img_w = img.shape[1]
        transform_matrix = transform_matrix_offset_center(zoom_matrix, img_h, img_w)
        #img = apply_affine_transform(img, zx=r, zy=r, channel_axis=2, fill_mode="constant", cval=1000)
        img = apply_transform(img, transform_matrix, 2, "constant", 1000)

        # 正解ボックスを変形させる
        w_rel = (boxes[:, [2]] - boxes[:, [0]]) / r
        h_rel = (boxes[:, [3]] - boxes[:, [1]]) / r
        dx = (boxes[:, [0]] - 0.5) * (1 - 1/r)
        dy = (boxes[:, [1]] - 0.5) * (1 - 1/r)
        boxes[:, [0]] -= dx
        boxes[:, [1]] -= dy
        boxes[:, [2]] = boxes[:, [0]] + w_rel
        boxes[:, [3]] = boxes[:, [1]] + h_rel

        for box in boxes:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            w = box[2] - box[0]
            h = box[3] - box[1]
            if (cx < 0.0 or 1.0 < cx or  cy < 0.0 or 1.0 < cy) or (w < 0.2 or h < 0.2):
                # 正解ボックスの中心が拡大後に無くなる場合や極端に小さくなる場合は変換前の画像を返す
                return original_img, original_boxes

        boxes = np.clip(boxes, 0.0, 1.0)

        #for box in boxes:
        #    coord = (box[0]*img_w, box[1]*img_h), (box[2]-box[0])*img_w, (box[3]-box[1])*img_h
        #    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        #    plt.imshow(img / 255.)
        #    currentAxis = plt.gca()
        #    currentAxis.add_patch(plt.Rectangle(*coord, fill=False, edgecolor=colors[3], linewidth=8))
        #    plt.savefig("zoom_sample.jpg")
        #    plt.clf()
        return img, boxes

    def random_shift(self, img, boxes):
        if self.shift_prob < np.random.random():
            return img, boxes

        original_img = np.copy(img)
        original_boxes = np.copy(boxes)

        img_h = img.shape[0]
        img_w = img.shape[1]
        tx = np.random.uniform(-self.shift_hrg, self.shift_hrg) * img_h
        ty = np.random.uniform(-self.shift_wrg, self.shift_wrg) * img_w
        dx = int(-ty)
        dy = int(-tx)
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        transform_matrix = translation_matrix
        #img = apply_affine_transform(img, tx=tx, ty=ty, channel_axis=2, fill_mode="nearest", cval=0)
        img = apply_transform(img, transform_matrix, 2, "nearest", 0)

        # 正解ボックスを変形させる
        dy /= img.shape[0]
        dx /= img.shape[1]
        boxes[:, [0, 2]] += dx
        boxes[:, [1, 3]] += dy
        boxes = np.clip(boxes, 0.0, 1.0)

        for box in boxes:
            w_rel = box[2] - box[0]
            h_rel = box[3] - box[1]
            if w_rel < self.min_box_size or h_rel < self.min_box_size:
                # 極端に正解ボックスが小さくなる場合は変換前の画像を返す
                return original_img, original_boxes

            # coord = (box[0]*img_w, box[1]*img_h), w_rel*img_w, h_rel*img_h
            # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            # plt.imshow(img / 255.)
            # currentAxis = plt.gca()
            # currentAxis.add_patch(plt.Rectangle(*coord, fill=False, edgecolor=colors[3], linewidth=8))
            # plt.savefig("shift_sample.jpg")
            # plt.clf()
        return img, boxes

    def random_sized_crop(self, img, boxes):
        if self.crop_prob < np.random.random():
            return img, boxes

        original_img = np.copy(img)
        original_boxes = np.copy(boxes)

        img_h = img.shape[0]
        img_w = img.shape[1]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.crop_aspect_range[1] -
                         self.crop_aspect_range[0])
        random_ratio += self.crop_aspect_range[0]

        # 面積を変えずに縦横の長さを変形
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        # 元の長さより長くなったら、元の長さに切り取る
        w = min(w, img_w)
        h = min(h, img_w)
        # 0-1の範囲に規格化した値
        w_rel = w / img_w
        h_rel = h / img_h
        # ピクセル単位の値
        w = int(w)
        h = int(h)

        # ↑で決めた縦横の幅ではみ出さないように、左上の座標を選ぶ
        x = np.random.random() * (img_w - w)
        y = np.random.random() * (img_h - h)
        # 0-1の範囲に規格化した値
        x_rel = x / img_w
        y_rel = y / img_h
        # ピクセル単位の値
        x = int(x)
        y = int(y)

        # 画像を切り抜く
        img = img[y:y + h, x:x + w]

        # 正解ボックスを変形させる
        new_boxes = []
        for box in boxes:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                # 正解ボックスの中心が切り抜き後も残っている
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_boxes.append(box)

                # coord = (xmin*w, ymin*h), (xmax-xmin)*w, (ymax-ymin)*h
                # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
                # plt.imshow(img / 255.)
                # currentAxis = plt.gca()
                # currentAxis.add_patch(plt.Rectangle(*coord, fill=False, edgecolor=colors[3], linewidth=8))
                # plt.savefig("crop_sample.jpg")
                # plt.clf()
            else:
                # 正解ボックスの中心が無くなってしまったら変形しないで元のデータを返す
                return original_img, original_boxes
        new_boxes = np.asarray(new_boxes).reshape(-1, boxes.shape[1])

        return img, new_boxes

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_file_paths)
                file_paths = self.train_file_paths
            else:
                file_paths = self.val_file_paths
            inputs = []
            targets = []
            for file_path in file_paths:
                #file_name = os.path.join(os.path.basename(os.path.dirname(file_path)), os.path.basename(file_path))
                file_name = os.path.basename(file_path)
                boxes = self.boxes[file_name]
                img, _, _ = load_img(file_path, target_size=input_shape)

                if train:
                    if self.blur_prob > 0:
                        img = self.blur(img)

                img = img_to_array(img)

                if train:
                    if self.zoom_prob > 0:
                        temp_img = np.copy(img)
                        temp_boxes = np.copy(boxes)
                        img, boxes = self.random_zoom(temp_img, temp_boxes)
                    if self.shift_prob > 0:
                        temp_img = np.copy(img)
                        temp_boxes = np.copy(boxes)
                        img, boxes = self.random_shift(temp_img, temp_boxes)
                    if self.crop_prob > 0:
                        temp_img = np.copy(img)
                        temp_boxes = np.copy(boxes)
                        img, boxes = self.random_sized_crop(temp_img, temp_boxes)
                img = imresize(img, self.image_size, interp='bicubic').astype('float32')

                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, boxes = self.horizontal_flip(img, boxes)

                boxes = self.bbox_util.assign_boxes(boxes)
                inputs.append(img)
                targets.append(boxes)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets
