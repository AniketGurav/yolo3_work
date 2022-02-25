import argparse
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phos_label_generator import gen_label
from phoc_label_generator import gen_phoc_label
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array,load_img



MODEL = "6_feb"
BATCH_SIZE = 64
EPOCHS = 10
LR = "1e-4"
train_csv_file = "./data/IAM_Data/IAM_train.csv"
valid_csv_file = "./data//IAM_Data//IAM_valid.csv"
train_unseen_csv_file = "./data/IAM_Data/IAM_valid.csv"
train_folder = "./data//IAM_Data/IAM_train//"
valid_folder = "./data/ZeroShot_Word_Recognition/ZSL_WordSpotting/IAM_Data/IAM_valid//"

print("\n\t str(BATCH_SIZE):", str(BATCH_SIZE), "\m model:", MODEL)
model_name = "new_" + MODEL + "_" + str(BATCH_SIZE) + "_"


class DataSequence(Sequence):
    def __init__(self, df, batch_size):
        self.df = df  # your pandas dataframe
        self.bsz = batch_size  # batch size

        # Take labels and a list of image locations in memory
        self.labels = []
        for i in range(len(self.df)):
            self.labels.append(
                {"phosnet": np.asarray(self.df['PhosLabel'].iloc[i]).astype(np.float32),
                 "phocnet": np.asarray(self.df['PhocLabel'].iloc[i]).astype(np.float32),
                 "text": np.asarray(self.df['Word'].iloc[i]).astype(np.string_)
                 })

        # print("\n\t labels:",self.labels)
        print("\n\t label length:", len(self.labels))
        self.im_list = self.df['Image'].tolist()

    def __len__(self):
        # compute number of batches to yield
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.im_list))
        self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx * self.bsz: (idx + 1) * self.bsz])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([img_to_array(load_img(im)) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        l1 = []
        l2 = []
        l3 = []
        for x in batch_y:
            l1.append(x['phosnet'])
            l2.append(x['phocnet'])
            l3.append(x['text'])
        # return batch_x, batch_y
        return batch_x, {'phosnet': np.asarray(l1), 'phocnet': np.asarray(l2), 'text': np.asarray(l3)}


def getphoclabel(x):
    return all_phoc_labels[x]


def getphoslabel(x):
    return all_phos_labels[x]


df_train = pd.read_csv(train_csv_file)

#print("\n\t df shape:", df_train.shape)
df_valid = pd.read_csv(valid_csv_file)
if train_unseen_csv_file != None:
    df_unseen = pd.read_csv(train_unseen_csv_file)
    df_train = df_train.merge(df_unseen, how='left', indicator=True)
    df_train = df_train[df_train['_merge'] == 'left_only']
    df_train = df_train[['Image', 'Word']]
if train_folder == valid_folder:
    df_train = df_train.merge(df_valid, how='left', indicator=True)
    df_train = df_train[df_train['_merge'] == 'left_only']
    df_train = df_train[['Image', 'Word']]

#print("Train_Images=", len(df_train), "Valid_Images=", len(df_valid))

# Generating dictionaries of words mapped to PHOS & PHOC vectors
train_word_phos_label = gen_label(list(set(df_train['Word'])))
valid_word_phos_label = gen_label(list(set(df_valid['Word'])))
all_phos_labels = {**train_word_phos_label, **valid_word_phos_label}
train_word_phoc_label = gen_phoc_label(list(set(df_train['Word'])))
valid_word_phoc_label = gen_phoc_label(list(set(df_valid['Word'])))
all_phoc_labels = {**train_word_phoc_label, **valid_word_phoc_label}

df_train['Image'] = train_folder + "/" + df_train['Image']
df_valid['Image'] = valid_folder + "/" + df_valid['Image']
df_train['PhosLabel'] = df_train['Word'].apply(getphoslabel)
df_valid['PhosLabel'] = df_valid['Word'].apply(getphoslabel)
df_train['PhocLabel'] = df_train['Word'].apply(getphoclabel)
df_valid['PhocLabel'] = df_valid['Word'].apply(getphoclabel)
df_train.to_csv("./data/temp2.csv",index=False)
