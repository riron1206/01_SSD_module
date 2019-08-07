"""
dtc_predict.py をjupyterでmoduleとして呼び出せるようにしたもの
20190118
"""
import os
import glob
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.misc import imread
from keras.preprocessing import image

import sys
sys.path.append(r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\AI_Edge_Contest\object_detection\SSD_classes_py\all_SSD_module\SSD')
#%matplotlib inline

from ssd_utils import BBoxUtility
#from ssd_vgg import num_classes, input_shape
from ssd_vgg import input_shape
from ssd_vgg import create_model, preprocess_input
from dtc_util_edit import load_img

from tqdm import tqdm
import cv2
import json

def predict_class_model(dst, model, img_height, img_width):
    """
    学習済み分類モデルで予測
    引数：
        dst:切り出したarray型の画像
        model:学習済み分類モデル
        img_height, img_width:モデルの入力画像サイズ（modelのデフォルトのサイズである必要あり）
    返り値：
        pred[pred_max_id]:確信度
        pred_max_id:予測ラベル
    """
    # 画像のサイズ変更
    x = cv2.resize(dst,(img_height,img_width))
    # 4次元テンソルへ変換
    x = np.expand_dims(x, axis=0)
    # 前処理
    X = x/255.0
    # 予測1画像だけ（複数したい場合は[0]をとる）
    pred = model.predict(X)[0]
    #print(pred)
    # 予測確率最大のクラスidを取得
    pred_max_id = np.argmax(pred)#,axis=1)
    return pred[pred_max_id], pred_max_id

def get_class_color(class_id):
    """
    クラスごとにBounding Boxの色をかえるための関数
    """
    class_id = int(class_id)
    if class_id%10==0:
        color = (255, 0, 0)
    elif class_id%10==1:
        color = (0, 255, 0)
    elif class_id%10==2:
        color = (0, 0, 255)
    elif class_id%10==3:
        color = (100, 255, 0)
    elif class_id%10==4:
        color = (100, 100, 0)
    elif class_id%10==5:
        color = (100, 100, 100)
    elif class_id%10==6:
        color = (50, 100, 100)
    elif class_id%10==7:
        color = (50, 100, 50)
    elif class_id%10==8:
        color = (50, 50, 100)
    elif class_id%10==9:
        color = (50, 50, 50)
    #print(type(color))
    # /255でタプルの各要素を割り算するためnumpyに変換
    color = np.array(color)
    # RGBA だと0-1の範囲内でないとダメみたい
    color = color/255
    #print(color)
    #print(type(color))
    # タプルに戻す
    color = tuple(color)
    #print(color)
    return color

def dtc_predict_py_edit(predict_dir, predicted_dir, dict, model_path
                        , conf_threshold=0.6, is_conf_threshold_down=False
                        , class_model=None
                        , dict_class={0.0:"Car", 1.0:"Bicycle", 2.0:"Pedestrian", 3.0:"Signal", 4.0:"Signs", 5.0:"Truck"}
                        , img_height=331, img_width=331
                        , is_overwrite=False
                        , max_box=100#None
                        , min_top_indices=0
                        , fontsize=4
                        , linewidth=0.5
                        ):
    """
    dtc_predict.py を一部変更した関数
    指定ディレクトリの画像1件ずつpredict実行し、バウンティングボックス付きの画像出力
    predictの位置や予測ラベルを書いたデータフレームも作成する
    Args:
        predict_dir : 予測したい画像がはいってるディレクトリ
        predicted_dir : 予測した画像出力先ディレクトリ
        dict : 予測クラスのidとクラス名の辞書型データ 例:dict = {0.0:"other", 1.0:"Bicycle", 2.0:"Pedestrian", 3.0:"Signal", 4.0:"Signs", 5.0:"Truck", 6.0:"Car"}
        model_path : ロードするモデルファイルのパス
        conf_threshold : 予測結果の確信度の閾値
        is_conf_threshold_down : 検出が出るまで予測結果の確信度の閾値を下げるかのフラグ
        class_model : 検出した領域をSSD以外のモデルで再予測する分類モデルオブジェクト
        dict_class : 再予測する分類モデルのクラスのidとクラス名の辞書型データ
        img_height, img_width : 再予測する分類モデルの入力画像サイズ（modelのデフォルトのサイズである必要あり）
        is_overwrite : 出力先に同名ファイルあればpredictしないかどうか
        max_box : 1画像で検出する領域の最大数。Noneなら制限なし。100なら100個まで検出
        min_top_indices : 最小でもmin_top_indices+1個は検出する。デフォルトの0なら最低1個は検出。is_conf_threshold_down=Trueでないと機能しない
        fontsize: 画像に表示する予測ラベルの文字の大きさ
        linewidth: 画像に表示する予測boxの線の太さ
    Return:
        なし（予測した画像出力、予測結果のデータフレーム出力(pred.csv)）
    """
    num_classes = len(dict)#6+1

    # 検出したとする確信度のしきい値
    #conf_threshold = 0.6#0.5#0.7

    # 予測する画像が入っているフォルダ
    #predict_dir = r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\AI_Edge_Contest\object_detection\SSD_classes_py\all_SSD_module\SSD\ssd_train'
    # 予測する画像のパス一覧
    img_path_list = glob.glob(os.path.join(predict_dir, "*.*"))

    # 予測結果を保存するフォルダ
    ##predicted_dir = r'D:\work\AI_Edge_Contest\object_detect\object_detection\SSD_classes\predicted_images'
    if not os.path.isdir(predicted_dir):
        os.mkdir(predicted_dir)

    file_names = []  # ファイル名一覧
    inputs = []      # ネットワークへ入力するため指定サイズに変形済みの画像データ
    images_h = []    # オリジナルサイズの画像の縦幅
    images_w = []    # オリジナルサイズの画像の横幅
    images = []      # 結果を見るためのオリジナルサイズの画像データ
    correctpred_filecount = 0

    # メニュー辞書作成
    #dict = {0.0:"other", 1.0:"Bicycle", 2.0:"Pedestrian", 3.0:"Signal", 4.0:"Signs", 5.0:"Truck", 6.0:"Car"}

    # モデルロード
    model = create_model(num_classes)
    #model.load_weights(r'D:\work\AI_Edge_Contest\object_detect\object_detection\SSD_classes\weight_ssd_best.hdf5')
    model.load_weights(model_path)
    print(model)

    import pandas as pd
    # 空のデータフレーム作成
    pred_df = pd.DataFrame(index=[], columns=['file_names', 'conf', 'label_name', 'x', 'y', 'x+w', 'y+h'])

    # ---- json用 ----
    prediction = {}
    # ----------------

    # 画像情報1件ずつ取得
    for path in tqdm(img_path_list):

        # 出力先に同名ファイルあればpredictしない
        if is_overwrite == False and os.path.isfile(os.path.join(predicted_dir, os.path.basename(path))):
            continue

        file_names = []
        file_names.append(os.path.basename(path))
        #print(file_names)
        # ---- json用 ----
        img_name = os.path.basename(path)
        prediction[img_name]={}
        # ----------------

        img, height, width = load_img(path, target_size=input_shape)
        img = image.img_to_array(img)

        inputs = []
        inputs.append(img.copy())

        images_h = []
        images_h.append(height)

        images_w = []
        images_w.append(width)

        images = []
        temp_image = imread(path)
        images.append(temp_image.copy())

        # 入力画像前処理
        inputs = preprocess_input(np.array(inputs))
        #print(inputs.shape)

        # 予測実行
        pred_results = model.predict(inputs, batch_size=1, verbose=0)
        #print(pred_results)
        bbox_util = BBoxUtility(num_classes)
        #print(bbox_util)
        bbox_results = bbox_util.detection_out(pred_results)
        #print(bbox_results)

        for file_no in range(len(file_names)):
        #for file_no in range(100):
            #print('-----------', file_names[file_no], '-----------')

            # 元の画像を描画
            plt.imshow(images[file_no] / 255.)

            # 予想したボックスの情報を取得
            bbox_label = bbox_results[file_no][:, 0]
            bbox_conf = bbox_results[file_no][:, 1]
            bbox_xmin = bbox_results[file_no][:, 2]
            bbox_ymin = bbox_results[file_no][:, 3]
            bbox_xmax = bbox_results[file_no][:, 4]
            bbox_ymax = bbox_results[file_no][:, 5]

            # 確信度がしきい値以上のボックスのみ抽出
            top_indices = [i for i, conf in enumerate(bbox_conf) if conf > conf_threshold]

            # --------- len(top_indices) > min_top_indices になるまでconf_threshold 下げるか --------------------
            if is_conf_threshold_down == True:
                conf_threshold_change = 0.0
                if len(top_indices) == 0:
                    # 基準のconf_threshold で検出なければ、検出でるまで閾値下げる
                    for conf_threshold_i in range(int(conf_threshold//0.01)):
                        conf_threshold_change = conf_threshold-((conf_threshold_i+1)*0.01)
                        top_indices = [i for i, conf in enumerate(bbox_conf) if conf > conf_threshold_change]
                        if len(top_indices) > min_top_indices:
                            #print('conf_threshold_i :', conf_threshold_i)
                            break
                            #continue
                #print('len(top_indices) :', len(top_indices))
                #print('conf_threshold_change :', conf_threshold_change)
            # -----------------------------------------------------------------------------------

            img_h = images_h[file_no]
            img_w = images_w[file_no]
            currentAxis = plt.gca()

            for box_no, top_index in enumerate(top_indices):
                # 検出数の最大値超えたらcontinue
                #（ AI_Edge_Contest では1画像に100件までの制限あるため）
                if (max_box is not None) and (box_no >= max_box):
                    continue

                # 予想したボックスを作成
                label = bbox_label[top_index]
                #print('label:', label)
                x = int(bbox_xmin[top_index]*img_w)
                y = int(bbox_ymin[top_index]*img_h)
                w = int((bbox_xmax[top_index]-bbox_xmin[top_index])*img_w)
                h = int((bbox_ymax[top_index]-bbox_ymin[top_index])*img_h)
                box = (x, y), w, h

                # 予想したボックスを描画
                conf = float(bbox_conf[top_index])
                label_name = dict[label]
                # -------------------- 分類モデルで予測 --------------------
                # 検出に加えるかのフラグ
                is_inclode = True
                if class_model is not None:
                    if conf < conf_threshold:
                        # (ndarray型の画像データから)検出領域切り出し
                        # ndarray型の切り出しは[y:y_max,x:x_max]の順番じゃないとおかしくなる
                        # https://qiita.com/tadOne/items/8967f046ca395669329d
                        tmp_img = images[file_no]
                        dst = tmp_img[y:y+h, x:x+w]
                        # ここで画像表示すると、bbox付き画像保存されない.あくまで確認用
                        #plt.imshow(dst / 255.)
                        #plt.show()

                        #print('file_names :', file_names[file_no])
                        #print('label_name :', label_name)
                        #print('conf :', conf)
                        # 切り出し画像を分類モデルでpredict
                        class_conf, class_label_id = predict_class_model(dst, class_model, img_height, img_width)
                        #print('class_label_name :', dict_class[class_label_id])
                        #print('class_conf :', class_conf)

                        # 分類モデルの方がスコア高ければ、ラベルとスコア書き換える
                        if conf <= class_conf:
                            label_name = dict_class[class_label_id]
                            conf = float(class_conf)
                        #elif top_index > 1:
                        #    # 検出数が1以上あってスコア低ければ検出に加えない
                        #    is_inclode = False
                # ---------------------------------------------------------

                # スコア低ければ検出に加えない
                if is_inclode == True:
                    # 画像にbbox描画
                    display_txt = '{:0.2f}, {}'.format(conf, label_name)
                    currentAxis.add_patch(plt.Rectangle(*box, fill=False, edgecolor=get_class_color(label), linewidth=linewidth))
                    currentAxis.text(x, y, display_txt, bbox={'facecolor': get_class_color(label), 'alpha': 0.2}, fontsize=fontsize)
                    # 結果をデータフレームで保持
                    series = pd.Series([file_names[file_no], conf, label_name, x, y, x+w, y+h], index=pred_df.columns)
                    #print(series)
                    pred_df = pred_df.append(series, ignore_index = True)
                    #print(pred_df)
                    # -------------------------- json用 --------------------------
                    if label_name not in prediction[img_name]:
                        prediction[img_name][label_name]=[]
                    prediction[img_name][label_name].append([x, y, x+w, y+h])
                    #print(prediction)
                    # ------------------------------------------------------------

            # 予測結果の画像ファイルを保存
            plt.savefig(os.path.join(predicted_dir, file_names[file_no]), dpi=300)
            plt.clf()

    output_dir = os.path.dirname(predicted_dir)
    pred_df.to_csv(os.path.join(output_dir, 'pred.csv'), sep='\t', index=False)

    # -------------------------- json用 --------------------------
    with open(os.path.join(output_dir, 'pred.json'), 'w') as f:
        json.dump(prediction, f, indent=4)# インデント付けてjsonファイル出力
    # ------------------------------------------------------------
