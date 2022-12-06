import datetime
import math
import os
from enum import Enum

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.models import Model, load_model
from keras.utils import to_categorical
from patchify import patchify
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
#from keras.layers import Rescaling
from tqdm import tqdm


def carregar_dividir_img(directory_path, patch_size):

    list = []

    for number, path in tqdm(enumerate(os.listdir(directory_path))):
        extension = path.split(".")[-1]
        if extension == "jpg" or extension == "png":

            caminho_img = rf"{directory_path}/{path}"

            image = cv2.imread(caminho_img)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            x = (image.shape[1] // patch_size) * patch_size  
            y = (image.shape[0] // patch_size) * patch_size  

            image = Image.fromarray(image)
            image = np.array(image.crop((0, 0, x, y)))
            patch_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

            for j in range(patch_img.shape[0]):
                for k in range(patch_img.shape[1]):
                    single_patch_img = patch_img[j, k]
                    list.append(np.squeeze(single_patch_img))

    return list


#def mudar_formas_img(instances):   
#    for j in range(len(instances)):
#       instances[j] = instances[j].reshape(-1, 1)
#    return instances


#def get_minimum_image_size(instances):
#    min_x = math.inf
#    min_y = math.inf
#    for image in instances:
#        min_x = image.shape[0] if image.shape[0] < min_x else min_x
#        min_y = image.shape[1] if image.shape[1] < min_y else min_y
#    return min_x, min_y


def mostrar_img(instances, rows=2, titles=None):
    n = len(instances)
    cols = n // rows if (n / rows) % rows == 0 else (n // rows) + 1

    for j, image in enumerate(instances):
        plt.subplot(rows, cols, j + 1)
        plt.title('') if titles is None else plt.title(titles[j])
        plt.axis("off")
        plt.imshow(image)

    plt.show()


def informacao_treinamento(root_directory):
    image_dataset, mask_dataset = [], []
    patch_size = 160
    for path, directories, files in os.walk(root_directory):
        for subdirectory in directories:
            if subdirectory == "images":
                image_dataset.extend(
                    carregar_dividir_img(os.path.join(path, subdirectory), patch_size=patch_size))
            elif subdirectory == "masks":
                mask_dataset.extend(
                    carregar_dividir_img(os.path.join(path, subdirectory), patch_size=patch_size))
    return np.array(image_dataset), np.array(mask_dataset)


#def segmentacao_binaria(image_dataset, mask_dataset):
#    x_reduced, y_reduced = [], []
#    for j, mask in tqdm(enumerate(mask_dataset)):
#        _img_height, _img_width, _img_channels = mask.shape
#        binary_image = np.zeros((_img_height, _img_width, 1)).astype(int)
#        for row in range(_img_height):
#            for col in range(_img_width):
#                rgb = mask[row, col, :]
#                binary_image[row, col] = 1 if rgb[0] == 60 and rgb[1] == 16 and rgb[2] == 152 else 0
#        if np.count_nonzero(binary_image == 1) > 0.15 * binary_image.size:
#            x_reduced.append(image_dataset[j])
#            y_reduced.append(binary_image)
#    return np.array(x_reduced), np.array(y_reduced)

class MaskColorMap(Enum):
    Unlabelled = (155, 155, 155)
    Building = (60, 16, 152)
    Land = (132, 41, 246)
    Road = (110, 193, 228)
    Vegetation = (254, 221, 58)
    Water = (226, 169, 41)


def mascara_encode_onehot(masks, num_classes):
    integer_encoded_labels = []

    for mask in tqdm(masks):
        _img_height, _img_width, _img_channels = mask.shape

        encoded_image = np.zeros((_img_height, _img_width, 1)).astype(int)

        for j, cls in enumerate(MaskColorMap):
            encoded_image[np.all(mask == cls.value, axis=-1)] = j

        integer_encoded_labels.append(encoded_image)

    return to_categorical(y=integer_encoded_labels, num_classes=num_classes)


dt_now = str(datetime.datetime.now()).replace(".", "_").replace(":", "_")
model_img_save_path = f"{os.getcwd()}/models/final_aerial_segmentation_{dt_now}.png"
model_save_path = f"{os.getcwd()}/models/final_aerial_segmentation_{dt_now}.hdf5"
model_checkpoint_filepath = os.getcwd() + "/models/weights-improvement-{epoch:02d}.hdf5"
csv_logger = rf"{os.getcwd()}/logs/aerial_segmentation_log_{dt_now}.csv"


def iou_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def jaccard_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


n_classes = 6

data_dir = r"semantic-segmentation-dataset"

X, Y = informacao_treinamento(root_directory=data_dir)

m, img_height, img_width, img_channels = X.shape
print('number of patched image training data:', m)

display_count = 6
random_index = [np.random.randint(0, m) for _ in range(display_count)]
sample_images = [x for z in zip(list(X[random_index]), list(Y[random_index])) for x in z]
mostrar_img(sample_images, rows=2)

Y = mascara_encode_onehot(Y, num_classes=n_classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


def build_unet(img_shape):
    inputs = Input(shape=img_shape)
    rescale = inputs
    previous_block_activation = rescale 
    contraction = {}

    for f in [16, 32, 64, 128]:
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            previous_block_activation)
        x = Dropout(0.1)(x)
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        contraction[f'conv{f}'] = x
        x = MaxPooling2D((2, 2))(x)
        previous_block_activation = x

    c5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
        previous_block_activation)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    previous_block_activation = c5

    for f in reversed([16, 32, 64, 128]):
        x = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(previous_block_activation)
        x = concatenate([x, contraction[f'conv{f}']])
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        previous_block_activation = x
        
    outputs = Conv2D(filters=n_classes, kernel_size=(1, 1), activation="softmax")(previous_block_activation)
    return Model(inputs=inputs, outputs=outputs)

model = build_unet(img_shape=(img_height, img_width, img_channels))
model.summary()
checkpoint = ModelCheckpoint(model_checkpoint_filepath, monitor="val_accuracy", verbose=1, save_best_only=True,
                             mode="max")
early_stopping = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="min")
csv_logger = CSVLogger(csv_logger, separator=",", append=False)
callbacks_list = [checkpoint, csv_logger] 
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy", iou_coefficient, jaccard_index])
model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), callbacks=callbacks_list,
          verbose=1)
model.save(model_save_path)
print("model saved:", model_save_path)
model_dir = '/Users/andrewdavies/Code/tensorflow-projects/u-net-aerial-imagery-segmentation/models/'
model_name = 'final_aerial_segmentation_2022-11-09 22_37_27_640199.hdf5'


def mascara_encode_rgb(mask):
    rgb_encode_image = np.zeros((mask.shape[0], mask.shape[1], 3))
    for j, cls in enumerate(MaskColorMap):
        rgb_encode_image[(mask == j)] = np.array(cls.value) / 255.
    return rgb_encode_image


for _ in range(20):
    test_img_number = np.random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = np.argmax(Y_test[test_img_number], axis=-1)
    test_img_input = np.expand_dims(test_img, 0)
    prediction = np.squeeze(model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis=-1)
    rgb_image = mascara_encode_rgb(predicted_img)
    rgb_ground_truth = mascara_encode_rgb(ground_truth)
    mostrar_img(
        [test_img, rgb_ground_truth, rgb_image],
        rows=1, titles=['Aerial', 'Ground Truth', 'Prediction']
    )