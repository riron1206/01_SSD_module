"""
dtc_train.py をjupyterでmoduleとして呼び出せるようにしたもの
20190118
"""
import os
import glob
import time

import sys
sys.path.append(r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\AI_Edge_Contest\object_detection\SSD_classes_py\all_SSD_module\SSD')

from dtc_util_edit import get_correct_boxes
from dtc_generator import Generator
#from ssd_vgg import num_classes, input_shape
from ssd_vgg import create_model, freeze_layers, create_prior_box
from ssd_vgg import input_shape
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.vis_utils import plot_model

def train_SSD300_NAG(master_file, train_dir, test_dir, model_path
                    , load_weights_path=r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\AI_Edge_Contest\object_detection\SSD_classes_py\all_SSD_module\SSD\weights_SSD300.hdf5'
                    , epochs=20 , batch_size=32, base_lr=1e-3
                    , num_classes=6+1
                    , callback=[]
                    ):
    """
    dtc_train.py のパラメータなどを引数にした関数
    ラベル情報のcsvファイルから訓練画像の領域情報ロードし、SSDのモデル作成する
    ※csvファイルからラベル情報読めるのが良いところ（一般的な物体検出モデルのラベル情報は1画像1xmlファイル）
    画像のサイズは300x300に変換される（ssd_vgg.pyより）
    分類器はVGG16のfine-tuning
    オプティマイザは ネステロフ+モメンタム+SGD(decayあり). 学習率はLearningRateScheduler でも下げる
    Args:
        master_file : 正解の座標（ファイル名, x, y, width, height, ラベルid）一覧のcsvファイルパス.
                      SSDの「背景」ラベルとして使われるため、ラベルidは0を使わないこと！！！
        train_dir : 訓練用画像が入っているフォルダパス
        test_dir : 評価用画像が入っているフォルダパス
        model_path : モデルファイルの保存先パス
        load_weights_path : 重みファイルのパス
        epochs : エポック数
        batch_size : バッチサイズ
        base_lr : 学習率初期値
        num_classes : クラス数。クラス数は「背景（class_id=0固定）」と「分類したいクラス」の数（要するにクラス数+1）にしないと正しくできない！！！！
        callback: 追加するcallbackのリスト。空なら ModelCheckpoint と LearningRateScheduler だけの callback にになる
    Return:
        なし（モデルファイルweight_ssd_best.hdf5 出力）
    """

    #epochs = 20        # エポック数
    #batch_size = 32     # バッチサイズ
    #base_lr =  1e-3     # 学習率初期値
    #num_classes = 11

    # 最適化関数
    # optimizer = keras.optimizers.Adam(lr=base_lr)
    # optimizer = keras.optimizers.RMSprop(lr=base_lr)
    optimizer = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=1e-6, nesterov=True)

    # 学習率のスケジュール関数
    def schedule(epoch, decay=0.90):
        return base_lr * decay**(epoch)

    # 正解の座標（ファイル名, x, y, width, height）一覧のcsvファイル
    #master_file = "xywh_train.csv"
    # 訓練用画像が入っているフォルダ
    #train_dir = "ssd_train"
    # 評価用画像が入っているフォルダ
    #test_dir = "ssd_test"

    # 画像ファイル名を指定すると正解座標が返ってくる辞書を作成
    correct_boxes = get_correct_boxes(master_file, train_dir, test_dir, num_classes=num_classes)

    # 画像ファイルパス一覧取得
    train_path_list = glob.glob(os.path.join(train_dir, "*.*"))
    test_path_list = glob.glob(os.path.join(test_dir, "*.*"))
    ## 画像ファイルパス一覧取得
    #train_path_list = []
    #test_path_list = []
    #for folder in glob.glob(os.path.join(train_dir, "*")):
    #    for file in glob.glob(os.path.join(folder, "*.jpg")):
    #        train_path_list.append(file)
    #for folder in glob.glob(os.path.join(test_dir, "*")):
    #    for file in glob.glob(os.path.join(folder, "*.jpg")):
    #        test_path_list.append(file)

    # モデル作成
    model = create_model(num_classes=num_classes)
    print('create_model ok')
    model.load_weights(load_weights_path, by_name=True)
    print('load_weights ok')

    # 入力付近の層をフリーズ
    freeze_layers(model, depth_level=1)
    print('freeze_layers ok')

    model.compile(optimizer=optimizer,
                  loss=MultiboxLoss(num_classes).compute_loss)
    #model.summary()
    plot_model(model, os.path.join(os.path.dirname(model_path), "model_ssd.png"))

    # デフォルトボックス作成
    priors = create_prior_box()

    # 画像データのジェネレータ作成
    bbox_util = BBoxUtility(num_classes, priors)
    gen = Generator(correct_boxes, bbox_util,
                    train_path_list, test_path_list,
                    (input_shape[0], input_shape[1]),
                    batch_size)

    print("Train Items : {}".format(gen.train_batches))
    print("Test  Items : {}".format(gen.val_batches))

    # コールバック設定
    callbacks = [ModelCheckpoint(model_path
                                , verbose=1,
                                 save_weights_only=True,
                                 save_best_only=True),
                 LearningRateScheduler(schedule)]
    if len(callback) != 0:
        callbacks.extend(callback)

    print(model.summary())

    # 学習開始
    start_time = time.time()
    history = model.fit_generator(gen.generate(True),
                        gen.train_batches//batch_size,
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=gen.generate(False),
                        validation_steps=gen.val_batches//batch_size)
    end_time = time.time()

    # 経過時間表示
    elapsed_time = end_time - start_time
    print("Elapsed Time : {0:d} hr {1:d} min {2:d} sec".
          format(int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60)))
    return history
