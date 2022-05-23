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


f = np.load(os.getcwd()+"/MusicFeatures.npz")#, spec= AllSpec, mel= AllMel, mfcc= AllMfcc, zcr= AllZcr, cen= AllCen, chroma= AllChroma, target=y)
S = f['spec']
mfcc = f['mfcc']
mel = f['mel']
chroma = f['chroma']
y = f['target']
print(S.shape)
print(mfcc.shape)
print(y.shape)

S_train, S_test, mfcc_train, mfcc_test, mel_train, mel_test, chroma_train, chroma_test, y_train, y_test = train_test_split(S, mfcc, mel, chroma, y, test_size= 0.2)


# MFCC
newtrain_mfcc = np.empty((mfcc_train.shape[0], 120, 600))
newtest_mfcc = np.empty((mfcc_test.shape[0], 120, 600))

for i in range(mfcc_train.shape[0]) :

    curr = mfcc_train[i]
    curr = cv2.resize(curr, (600, 120))
    newtrain_mfcc[i] = curr

mfcc_train = newtrain_mfcc

for i in range(mfcc_test.shape[0]) :

    curr = mfcc_test[i]
    curr = cv2.resize(curr, (600, 120))
    newtest_mfcc[i] = curr

mfcc_test = newtest_mfcc



maximum = np.amax(mfcc_train)
# mfcc_train = mfcc_train/maximum
# mfcc_test = mfcc_test/maximum

mfcc_train = mfcc_train.astype(np.float32)
mfcc_test = mfcc_test.astype(np.float32)


N, row, col = mfcc_train.shape
mfcc_train = mfcc_train.reshape((N, row, col, 1))

N, row, col = mfcc_test.shape
mfcc_test = mfcc_test.reshape((N, row, col, 1))
# print(mfcc_train.shape, mfcc_test.shape)


mean_data = np.mean(mfcc_train)
std_data = np.std(mfcc_train)

mfcc_train = (mfcc_train - mean_data)/ std_data
mfcc_test = (mfcc_test - mean_data)/ std_data

# print(np.amin(mfcc_train), np.amax(mfcc_train))
# print(np.amin(mfcc_test), np.amax(mfcc_test))

# Spectrogram
maximum1 = np.amax(S_train)
S_train = S_train/np.amax(maximum1)
S_test = S_test/np.amax(maximum1)

S_train = S_train.astype(np.float32)
S_test = S_test.astype(np.float32)

N, row, col = S_train.shape
S_train = S_train.reshape((N, row, col, 1))

N, row, col = S_test.shape
S_test = S_test.reshape((N, row, col, 1))
# print(S_train.shape, S_test.shape)


# Mel-Spectrogram

maximum = np.amax(mel_train)
mel_train = mel_train/np.amax(maximum)
mel_test = mel_test/np.amax(maximum)

mel_train = mel_train.astype(np.float32)
mel_test = mel_test.astype(np.float32)

N, row, col = mel_train.shape
mel_train = mel_train.reshape((N, row, col, 1))

N, row, col = mel_test.shape
mel_test = mel_test.reshape((N, row, col, 1))
# print(mel_train.shape, mel_test.shape)


# Chromagram

newchroma_train = np.empty((chroma_train.shape[0], 120, 800))
for i in range(chroma_train.shape[0]) :

    curr = chroma_train[i]
    curr = cv2.resize(curr, (800, 120))
    newchroma_train[i] = curr

chroma_train = newchroma_train
# X = X.transpose(0,2,1)

newchroma_test = np.empty((chroma_test.shape[0], 120, 800))
for i in range(chroma_test.shape[0]) :

    curr = chroma_test[i]
    curr = cv2.resize(curr, (800, 120))
    newchroma_test[i] = curr

chroma_test = newchroma_test

maximum = np.amax(chroma_train)
# X_train = X_train/maximum
# X_test = X_test/maximum

chroma_train = chroma_train.astype(np.float32)
chroma_test = chroma_test.astype(np.float32)


N, row, col = chroma_train.shape
chroma_train = chroma_train.reshape((N, row, col, 1))

N, row, col = chroma_test.shape
chroma_test = chroma_test.reshape((N, row, col, 1))
print(chroma_train.shape, chroma_test.shape)

print(chroma_train.shape, chroma_test.shape)

mean_data = np.mean(chroma_train)
std_data = np.std(chroma_train)

chroma_train = (chroma_train - mean_data)/ std_data
chroma_test = (chroma_test - mean_data)/ std_data


# Save Spectrogram train-test
np.savez_compressed(os.getcwd()+"/new_spectrogram_train_test.npz", S_train= S_train, S_test= S_test, y_train = y_train, y_test= y_test)

# Save MFCC train-test
np.savez_compressed(os.getcwd()+"/new_mfcc_train_test.npz", mfcc_train= mfcc_train, mfcc_test= mfcc_test, y_train = y_train, y_test= y_test)

# Save Mel-Spectrogram train-test
np.savez_compressed(os.getcwd()+"/new_mel_train_test.npz", mel_train= mel_train, mel_test= mel_test, y_train = y_train, y_test= y_test)

# Save Chromagram train-test
np.savez_compressed(os.getcwd()+"/new_chroma_train_test.npz", chroma_train= chroma_train, chroma_test= chroma_test, y_train = y_train, y_test= y_test)


# Load Spectrogram Train-test data

spec_file = np.load(os.getcwd()+"/new_spectrogram_train_test.npz")



# Model 1 for Spectrogram
S_train = spec_file['S_train']
S_test = spec_file['S_test']
y_train = spec_file['y_train']
y_test = spec_file['y_test']

model = Sequential()
model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= S_train[0].shape, padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')

model.summary()

# Train Model 1

checkpoint = ModelCheckpoint(os.getcwd()+"/models/new_spec_model_spectrogram1_{epoch:03d}.h5", period= 5)

model.fit(S_train, y_train, epochs= 100, callbacks= [checkpoint], batch_size= 32, verbose= 1)
model.save(os.getcwd() + "/models/new_spec_model_spectrogram1.h5")

model = load_model(os.getcwd() + "/models/new_model_spectrogram1.h5")


# Training Accuracy
y_pred = model.predict(S_train)
y_pred = np.argmax(y_pred, axis= -1)
y_true = np.argmax(y_train, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100

print("Train Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")



# Testing Accuracy
y_pred = model.predict(S_test)
y_pred = np.argmax(y_pred, axis= -1)
y_true = np.argmax(y_test, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100

print("Test Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

class_names = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
conf_mat = confusion_matrix(y_true, y_pred, normalize= 'true')
conf_mat = np.round(conf_mat, 2)

conf_mat_df = pd.DataFrame(conf_mat, columns= class_names, index= class_names)

plt.figure(figsize = (10,7), dpi = 200)
sn.set(font_scale=1.4)
sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16}) # font size
plt.tight_layout()
plt.savefig(os.getcwd() + "/new_spec_conf_mat1.png")

####################################################################################

# Spectrogram Model 2

S_train = spec_file['S_train']
S_test = spec_file['S_test']
y_train = spec_file['y_train']
y_test = spec_file['y_test']

model = Sequential()
model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= S_train[0].shape, padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')

model.summary()

# Train Model 2

checkpoint = ModelCheckpoint(os.getcwd()+"/models/new_spec_model_spectrogram2_{epoch:03d}.h5", period= 5)

model.fit(S_train, y_train, epochs= 100, callbacks= [checkpoint], batch_size= 32, verbose= 1)
model.save(os.getcwd() + "/models/new_spec_model_spectrogram2.h5")

model = load_model(os.getcwd() + "/models/new_model_spectrogram2.h5")


# Training Accuracy
y_pred = model.predict(S_train)
y_pred = np.argmax(y_pred, axis= -1)
y_true = np.argmax(y_train, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100

print("Train Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

# Training Accuracy
y_pred = model.predict(S_test)
y_pred = np.argmax(y_pred, axis= -1)
y_true = np.argmax(y_train, axis= -1)

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
plt.savefig(os.getcwd() + "/new_spec_conf_mat2.png")


##################################################################################################################################################
##################################################################################################################################################

# MFCC

mfcc_file = np.load(os.getcwd()+"/new_mfcc_train_test.npz")
mfcc_train = mfcc_file['mfcc_train']
mfcc_test = mfcc_file['mfcc_test']
y_train = mfcc_file['y_train']
y_test = mfcc_file['y_test']


def get_model() :

    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape= mfcc_train[0].shape, activation= 'tanh', padding= 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4,6), padding= 'same'))
    model.add(Conv2D(32, (3,3), input_shape= mfcc_train[0].shape, activation= 'tanh', padding= 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4,6), padding= 'same'))
    model.add(Conv2D(64, (3,3), input_shape= mfcc_train[0].shape, activation= 'tanh', padding= 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4,6), padding= 'same'))
    model.add(Flatten())
    # model.add(Dense(256, activation= 'tanh'))
    model.add(Dense(256, activation= 'tanh'))
    model.add(Dense(64, activation= 'tanh'))
    model.add(Dense(10, activation= 'softmax'))

    model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')

    model.summary()

    return model



def get_majority(pred) :

    pred = [ [1,2,3],  [1,1,3], [1,1,2] ]



    N = len(pred[0])
    vote = []
    for i in range(N) :
        candidates = [x[i] for x in pred]
        candidates = np.array(candidates)
        uniq, freq = np.unique(candidates, return_counts= True)
        vote.append(uniq[np.argmax(freq)])

    vote = np.array(vote)

    return vote


# Train Model 1

model1 = get_model()

kf = KFold(n_splits = 10)
for train_index, val_index in kf.split(mfcc_train, np.argmax(y_train, axis= -1)):

    kf_mfcc_train = mfcc_train[train_index]
    kf_X_val = mfcc_train[val_index]
    kf_y_train = y_train[train_index]
    kf_y_val = y_train[val_index]

    model1.fit(kf_mfcc_train, kf_y_train, validation_data= (kf_X_val, kf_y_val), epochs= 30, batch_size= 30, verbose= 1)
    model1.save(os.getcwd() + "/models/new_ensemble_mfcc1.h5")



# Train Model 2

model2 = get_model()

kf = KFold(n_splits = 10)
for train_index, val_index in kf.split(mfcc_train, np.argmax(y_train, axis= -1)):

    kf_mfcc_train = mfcc_train[train_index]
    kf_X_val = mfcc_train[val_index]
    kf_y_train = y_train[train_index]
    kf_y_val = y_train[val_index]

    model2.fit(kf_mfcc_train, kf_y_train, validation_data= (kf_X_val, kf_y_val), epochs= 30, batch_size= 30, verbose= 1)
    model2.save(os.getcwd() + "/models/new_ensemble_mfcc2.h5")



# Train Model 3

model3 = get_model()

kf = KFold(n_splits = 10)
for train_index, val_index in kf.split(mfcc_train, np.argmax(y_train, axis= -1)):

    kf_mfcc_train = mfcc_train[train_index]
    kf_X_val = mfcc_train[val_index]
    kf_y_train = y_train[train_index]
    kf_y_val = y_train[val_index]

    model3.fit(kf_mfcc_train, kf_y_train, validation_data= (kf_X_val, kf_y_val), epochs= 30, batch_size= 30, verbose= 1)
    model3.save(os.getcwd() + "/models/new_ensemble_mfcc3.h5")

model1 = load_model(os.getcwd() + "/models/new_ensemble_mfcc1.h5")
model2 = load_model(os.getcwd() + "/models/new_ensemble_mfcc2.h5")
model3 = load_model(os.getcwd() + "/models/new_ensemble_mfcc3.h5")

# Training Accuracy
y_true = np.argmax(y_train, axis= -1)

y_pred1 = model1.predict(mfcc_train)
y_pred1 = np.argmax(y_pred1, axis= -1)

y_pred2 = model2.predict(mfcc_train)
y_pred2 = np.argmax(y_pred2, axis= -1)

y_pred3 = model3.predict(mfcc_train)
y_pred3 = np.argmax(y_pred3, axis= -1)

y_pred = [y_pred1, y_pred2, y_pred3]

y_pred = get_majority(y_pred)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100

print("Train Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")


# Test Model


y_true = np.argmax(y_test, axis= -1)

y_pred1 = model1.predict(mfcc_test)
y_pred1 = np.argmax(y_pred1, axis= -1)

y_pred2 = model2.predict(mfcc_test)
y_pred2 = np.argmax(y_pred2, axis= -1)

y_pred3 = model3.predict(mfcc_test)
y_pred3 = np.argmax(y_pred3, axis= -1)

y_pred = [y_pred1, y_pred2, y_pred3]

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
plt.savefig(os.getcwd() + "/new_ensemble_mfcc_conf_mat.png")



##################################################################################################################################################
##################################################################################################################################################

# Mel-Sprectrogram

file = np.load(os.getcwd()+"/new_mel_train_test.npz")
mel_train = file['mel_train']
mel_test = file['mel_test']
y_train = file['y_train']
y_test = file['y_test']

# print(X_train.shape)
# sys.exit(1)
model = Sequential()
model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= mel_train[0].shape, padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Flatten())
model.add(Dense(64, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')

model.summary()


# Train Model

checkpoint = ModelCheckpoint(os.getcwd()+"/models/ensemble_model_melspectrogram1_{epoch:03d}.h5", period= 5)

model.fit(mel_train, y_train, epochs= 200, callbacks= [checkpoint], batch_size= 32, verbose= 1)
model.save(os.getcwd() + "/models/ensemble_model_melspectrogram1.h5")


model = load_model(os.getcwd() + "/models/ensemble_model_melspectrogram1.h5")


# Training Accuracy
y_pred = model.predict(mel_train)
y_pred = np.argmax(y_pred, axis= -1)
y_true = np.argmax(y_train, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100

print("Train Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

# Testing Accuracy
y_pred = model.predict(mel_test)
y_pred = np.argmax(y_pred, axis= -1)
y_true = np.argmax(y_test, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100
print("Testing Accuracy", acc)

class_names = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
conf_mat = confusion_matrix(y_true, y_pred, normalize= 'true')
conf_mat = np.round(conf_mat, 2)

conf_mat_df = pd.DataFrame(conf_mat, columns= class_names, index= class_names)

plt.figure(figsize = (10,7), dpi = 200)
sn.set(font_scale=1.4)
sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16}) # font size
plt.tight_layout()
plt.savefig(os.getcwd() + "/ensemble_mel_conf_mat1.png")
