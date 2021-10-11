import pandas as pd
import csv
import tensorflow as tf
import numpy as np
import sys
import cv2

# 再現性
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow_addons as tfa
from vit_keras import vit, utils
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard

def build_VIT():
  # パッチサイズ16
  vit16 = vit.vit_b16(
    image_size = size,
    activation = "softmax",
    pretrained = True,
    include_top = False,
    pretrained_top = False
  )
  
  model = tf.keras.Sequential([
    vit16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(21, activation = tfa.activations.gelu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(num_classes, 'softmax')
    ],
    name = 'vision_transformer')
     
  model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
  return model


def data_preprocessing():
#　Data Preprocessing
  img_list = []
  for i in range(len(df)):
    img_list.append(cv2.resize(cv2.imread("../../"+df.PATH[i]),(size,size)))
      
  x_train, x_test, y_train, y_test = train_test_split(img_list, y, test_size=0.2,random_state=1)
  x_test,x_val,y_test,y_val = train_test_split(x_test, y_test, test_size=0.5,random_state=1)
  x_train,x_val,x_test = np.array(x_train),np.array(x_val),np.array(x_test)
  y_train,y_test,y_val = to_categorical(np.array(y_train).reshape(-1,1)),to_categorical(np.array(y_test).reshape(-1,1)),to_categorical(np.array(y_val).reshape(-1,1))
  return x_train,x_val,x_test,y_train,y_val,y_test

if __name__ == "__main__":
  # Loading the data
  args = sys.argv
  # K Size args[1],Image Size args[2] num_classes args[3]
  size = int(args[2])
  data_path = "../../data/data" + args[3] + "size" + args[1]+ ".csv"
  df = pd.read_csv(data_path)
  y = df['class']

  print(df)

  x_train,x_val,x_test,y_train,y_val,y_test = data_preprocessing()
  # 定義
  input_shape = (int(args[2]),int(args[2]),3)
  num_classes = args[3]
  BS = 128
  stepsEpoch = int(df.shape[0]/BS)
  epochs = 100 
  print(stepsEpoch)


  # model build
  model = build_VIT()
  model.summary()
  history = model.fit(x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_data=(x_val,y_val)
    )
  score = model.evaluate(x_test, y_test, verbose=0,batch_size=128)
  res = pd.DataFrame(score)
  res.to_csv("resultVision20_64.csv")

