import os, sys, cv2
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential, Input,Model, load_model
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, MaxPool1D, GaussianNoise, GlobalMaxPooling1D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
import tensorflow
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

spec_file = np.load(os.getcwd()+"/new_spectrogram_train_test.npz")

# Model 1 for Spectrogram
S_train = spec_file['S_train']
S_test = spec_file['S_test']
y_train = spec_file['y_train']
y_test = spec_file['y_test']

model1 = load_model(os.getcwd() + "/models/new_spec_model_spectrogram1.h5")

####################################################################################

# Spectrogram Model 2

S_train = spec_file['S_train']
S_test = spec_file['S_test']
y_train = spec_file['y_train']
y_test = spec_file['y_test']


model2 = load_model(os.getcwd() + "/models/new_spec_model_spectrogram2.h5")


##################################################################################################################################################
##################################################################################################################################################

# MFCC

mfcc_file = np.load(os.getcwd()+"/new_mfcc_train_test.npz")
mfcc_train = mfcc_file['mfcc_train']
mfcc_test = mfcc_file['mfcc_test']
y_train = mfcc_file['y_train']
y_test = mfcc_file['y_test']

model3 = load_model(os.getcwd() + "/models/new_ensemble_mfcc1.h5")
model4 = load_model(os.getcwd() + "/models/new_ensemble_mfcc2.h5")
model5 = load_model(os.getcwd() + "/models/new_ensemble_mfcc3.h5")

##################################################################################################################################################
##################################################################################################################################################

# Mel-Sprectrogram

file = np.load(os.getcwd()+"/new_mel_train_test.npz")
mel_train = file['mel_train']
mel_test = file['mel_test']
y_train = file['y_train']
y_test = file['y_test']

model6 = load_model(os.getcwd() + "/models/ensemble_model_melspectrogram1.h5")

##################################################################################################################################################
##################################################################################################################################################

# Ensemble


y_true = np.argmax(y_train, axis= -1)

y_pred1 = model1.predict(S_train)
y_pred1 = np.argmax(y_pred1, axis= -1)

y_pred2 = model2.predict(S_train)
y_pred2 = np.argmax(y_pred2, axis= -1)

y_pred3 = model3.predict(mfcc_train)
y_pred3 = np.argmax(y_pred3, axis= -1)

y_pred4 = model4.predict(mfcc_train)
y_pred4 = np.argmax(y_pred4, axis= -1)

y_pred5 = model5.predict(mfcc_train)
y_pred5 = np.argmax(y_pred5, axis= -1)

y_pred6 = model6.predict(mel_train)
y_pred6 = np.argmax(y_pred6, axis= -1)

y_pred = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
y_pred = get_majority(y_pred)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100

print("Training Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")
# Test Model
y_true = np.argmax(y_test, axis= -1)

y_pred1 = model1.predict(S_test)
y_pred1 = np.argmax(y_pred1, axis= -1)

y_pred2 = model2.predict(S_test)
y_pred2 = np.argmax(y_pred2, axis= -1)

y_pred3 = model3.predict(mfcc_test)
y_pred3 = np.argmax(y_pred3, axis= -1)

y_pred4 = model4.predict(mfcc_test)
y_pred4 = np.argmax(y_pred4, axis= -1)

y_pred5 = model5.predict(mfcc_test)
y_pred5 = np.argmax(y_pred5, axis= -1)

y_pred6 = model6.predict(mel_test)
y_pred6 = np.argmax(y_pred6, axis= -1)

y_pred = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
y_pred = get_majority(y_pred)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100
print("Testing Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

class_names = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
conf_mat = confusion_matrix(y_true, y_pred, normalize= 'true')
conf_mat = np.round(conf_mat, 2)

conf_mat_df = pd.DataFrame(conf_mat, columns= class_names, index= class_names)

plt.figure(figsize = (10,7), dpi = 200)
sn.set(font_scale=1.4)
sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16}) # font size
plt.tight_layout()
plt.savefig(os.getcwd() + "/new_ensemble_conf_mat.png")


