import os
import glob
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.utils import to_categorical
from random import randint
import pandas as pd
import pickle
import ast
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

f = open("miniclasses.txt","r")
classes = f.readlines()
f.close()
classes = [c.replace('\n','').replace(' ','_') for c in classes]
n = len(classes)
print('Number of classes: ', n)
vals = [x for x in range(0,n)]
classmap = dict(zip(classes, vals))
classmaprev = dict(zip(vals,classes))

all_files = glob.glob(os.path.join('shuffletrain4/', '*.csv'))
batchsize = 100
N_CLASSES = 340
STEPS = 800
EPOCHS = 500
SIZE = 256
TEST_DIR = ''
N_FILES = 100

def draw_cv2(raw_strokes, size, lw=6, time_color=True):
    img = np.zeros((256, 256), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != 256:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator_xd(all_files,classmap, size, batchsize,ks,N_CLASSES, lw=6, time_color=True):
    while True:
        fn = 0
        for k in np.random.permutation(ks):
            print('Starting file: ', fn)
            fn += 1
            filename = all_files[k]
            # num_skipped = 0
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                """
                num_recognized = 0
                for r in df['recognized'].values:
                  if r:
                    num_recognized += 1
                  else:
                    num_skipped += 1
                x = np.zeros((num_recognized, size, size, 1))
                """
                x = np.zeros((len(df), size, size, 1))
                labels = []
                i = 0
                for j, raw_strokes in enumerate(df.drawing.values):
                  # if df['recognized'].values[j]:
                  x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                           time_color=time_color)
                  labels.append(df['word'].values[j])
                  i += 1
                labels = [c.replace(' ','_') for c in labels]
                idx = [classmap[m] for m in labels]
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(idx, N_CLASSES)
                yield x, y
            # print('Number skipped for file: ', num_skipped)
        print('Done with all files. Restarting')

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

# valid_df = pd.read_csv(os.path.join('sonam_shuffled/shuffledtrain2/', 'shuffledtrain2_{}.csv'.format(N_FILES - 1)), nrows=34000)
valid_df = pd.read_csv(os.path.join('shuffletrain4/', 'shuffledtrain_{}.csv'.format(N_FILES - 1)), nrows=34000)
x_valid = df_to_image_array_xd(valid_df, SIZE)
labels = valid_df['word']
labels = [lc.replace(' ', '_') for lc in labels]
idx = [classmap[lx] for lx in labels]
y_valid = keras.utils.to_categorical(idx, num_classes=N_CLASSES)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

train_datagen = image_generator_xd(all_files, classmap, size=SIZE, N_CLASSES=N_CLASSES, batchsize=batchsize, ks=range(N_FILES - 1))

"""
x, y = next(train_datagen)
print(x.shape)
print(y.shape)
"""

model = MobileNet(input_shape=(SIZE, SIZE, 1), alpha=1., weights=None, classes=N_CLASSES)
model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())

callbacks = [
    ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.75, patience=3,
                      min_delta=0.001, mode='max', min_lr=1e-5, verbose=1),
    ModelCheckpoint('beluga_model_256.h5', monitor='val_top_3_accuracy', mode='max', save_best_only=True,
                    save_weights_only=True),
]
hists = []
hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)
hists.append(hist)

test = pd.read_csv('../test_simplified.csv')
test.head()
x_test = df_to_image_array_xd(test, SIZE)
print(test.shape, x_test.shape)
print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))

test_predictions = model.predict(x_test, batch_size=128, verbose=1)
top3 = preds2catids(test_predictions)
print(top3.head())
print(top3.shape)

top3cats = top3.replace(classmaprev)
print(top3cats.head())
print(top3cats.shape)

test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('submission_test.csv', index=False)
print(submission.head())
print(submission.shape)
