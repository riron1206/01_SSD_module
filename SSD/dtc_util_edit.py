import os
import codecs
import csv
import numpy as np
from scipy.misc import imread
from PIL import Image
import keras


def get_correct_boxes(master_file, train_dir, test_dir, num_classes=6):
    """
    画像ファイル名を指定すると正解座標が返ってくる辞書を作成

    :param master_file: ファイル名,x,y,w,h が書かれたcsvファイル
    :param train_dir: 訓練用画像が入っているフォルダ名
    :param test_dir: テスト用画像が入っているフォルダ名
    :param num_classes: to_categorical()でワンホットラベルにするための総クラス数
    :return: 画像ファイル名を指定すると正解座標が返ってくる辞書
    """
    boxes = {}
    with codecs.open(master_file, "r", "shift-jis") as f:
        reader = csv.reader(f)
        prev_key = None
        box_one_pict = []
        for row in reader:
            file_name = row[0]
            # 画像の縦横サイズを取得する
            train_file_path = os.path.join(train_dir, file_name)
            test_file_path = os.path.join(test_dir, file_name)
            #print(train_file_path)
            #print(test_file_path)
            if os.path.exists(train_file_path):
                img = imread(train_file_path)
            elif os.path.exists(test_file_path):
                img = imread(test_file_path)
            else:
                assert False, "error"
            max_width = img.shape[1]
            max_height = img.shape[0]
            # 答えの座標にマイナスがある場合は0にしておく
            box = [0 if int(i) < 0 else int(i) for i in row[1:5]]
            # width または height が0のものはスキップ
            if box[2] == 0 or box[3] == 0:
                continue
            # widthとheightから右下の座標を計算
            box[2] += box[0]
            box[3] += box[1]
            # 0.0-1.0の範囲に正規化する
            box[0] = max(0.0, min(box[0] / max_width, 1.0))
            box[1] = max(0.0, min(box[1] / max_height, 1.0))
            box[2] = max(0.0, min(box[2] / max_width, 1.0))
            box[3] = max(0.0, min(box[3] / max_height, 1.0))

            if num_classes == 2:
                # 正解座標の後ろに正解クラスをone-hot表現で追加（今回は「手」の1種類なので全て1）
                box.append(1)
            else:
                # クラスベクトル（0からnb_classesまでの整数）をcategorical_crossentropyとともに用いるためのバイナリのクラス行列に変換
                box.extend(keras.utils.to_categorical(int(row[5])-1, num_classes-1))#6))#
                #box.extend(keras.utils.to_categorical(int(row[5]), num_classes))

            # 別ファイルに変わったらキーを変更
            if prev_key is None:
                prev_key = file_name
            elif prev_key != file_name:
                boxes[prev_key] = np.array(box_one_pict)
                box_one_pict = []
                prev_key = file_name
            box_one_pict.append(box)

        boxes[prev_key] = np.array(box_one_pict)
    return boxes


def load_img(path, target_size=None):
    img = Image.open(path)
    height = img.height
    width = img.width
    img = img.convert('RGB')
    if target_size:
        NEAREST = NONE = 0
        LANCZOS = ANTIALIAS = 1
        BILINEAR = LINEAR = 2
        BICUBIC = CUBIC = 3
        img = img.resize((target_size[1], target_size[0]), resample=BICUBIC)
    return img, height, width
