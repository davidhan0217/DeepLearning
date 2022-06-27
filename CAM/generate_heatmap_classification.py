import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import datetime
from sklearn.model_selection import GroupShuffleSplit

AUTOTUNE = tf.data.experimental.AUTOTUNE

gpus = tf.config.experimental.list_physical_devices('CPU')
tf.config.experimental.set_memory_growth(gpus, True)
# tf.config.experimental.set_memory_growth(gpus[1], True)

heatMapSavePath = "/home/monster/Documents/AREDS/HeatMap/"
checkpoint_path = os.path.join(heatMapSavePath, "HeatmapAnalysis", "WeightFile",
                               "Weight", "weights")
processedHeatMapSavePath = os.path.join(heatMapSavePath, "HeatmapAnalysis", "ProcessResult",
                                        "1626510116_123_INCEPTION_batch8_1024x1024")

image_size = 1024
image_channel = 3
mini_batch_size = 12

base_image_dir = os.path.join("/media/monster/Storage/AREDS/AREDS gb1024 with mask")
label_dir = os.path.join("/home/monster/Documents/AREDS/LabelFiles/RemovalOfSwitchingResults3rdJune2021_ForTraining.csv")

col_list = ["image_name", "Patient_ID", "SYS1", "SYS2", "DIA1", "DIA2", "Class"]
image_name = "image_name"
label_name = "Class"
group_name = "patient_ID"

retina_df = pd.read_csv(label_dir, usecols=col_list)
retina_df = retina_df.dropna()
retina_df.reset_index()
retina_df['Patient_ID'] = retina_df['Patient_ID'].astype(str)
retina_df['Class'] = retina_df['Class'].astype(str)

export_df = pd.DataFrame([['', 0, 0, 0]], columns=['file_name', 'true_result', 'predict_result', 'confidence_score'])

# retina_df["SYS2"] = retina_df["SYS2"] / 250

train_index, test_index = next(
    GroupShuffleSplit(test_size=.20, n_splits=2, random_state=10).split(retina_df, groups=retina_df['Patient_ID']))
train = retina_df.iloc[train_index]
test = retina_df.iloc[test_index]

val_index, test_index = next(
    GroupShuffleSplit(test_size=.50, n_splits=2, random_state=10).split(test, groups=test['Patient_ID']))
validation = test.iloc[val_index]
test = test.iloc[test_index]

train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)


print('TRAIN dataset:')
print(train.groupby(label_name).size())
# print("plus {} extra mecular unhealthy image".format(len(STARE_train_df_one_label) + 88))  # count = 88
print('')
print('VALIDATION dataset:')
print(validation.groupby(label_name).size())
print('')
print('TEST dataset:')
print(test.groupby(label_name).size())
print('')

print('Starting pre-processing for data.Dataset at {}'.format(datetime.datetime.now().time()))


mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"], #devices=["/gpu:0", "/gpu:1"],
                                                   cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with mirrored_strategy.scope():
    inputs = tf.keras.Input(shape=(image_size, image_size, image_channel))
    model = tf.keras.applications.InceptionResNetV2(include_top=False, input_tensor=inputs, weights='imagenet')
    # model = efn.EfficientNetB2(include_top=False, input_tensor=inputs, weights='imagenet')
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    top_dropout_rate = 0.5
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="InceptionResnetV2")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.000001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()
                           ])

    model.load_weights(checkpoint_path)

print(model.summary())
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

last_conv_layer_name = "conv_7b_ac"
classifier_layer_names = [
    "avg_pool",
    "predictions",
]


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)

    # Important to have this here, this would have the rescale if model does not have it.
    array = array / 255
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    # grads = grads / (grads.numpy().max() - grads.numpy().min())

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


preprocess_input = tf.keras.applications.efficientnet.preprocess_input


def generate_heat_map(fileName, trueValue, srcPath, dstPath, exp_df):
    img_path = os.path.join(srcPath, fileName)
    test_img = cv2.imread(img_path)
    if test_img is not None:

        img_array = preprocess_input(get_img_array(img_path, size=(image_size, image_size)))

        # data = [{"image_name": fileName, "SYS2": trueValue}]
        # df = pd.DataFrame(data)
        # datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        # val_ds = datagen.flow_from_dataframe(dataframe=df,
        #                                      directory=base_image_dir,
        #                                      x_col=image_name,
        #                                      y_col=label_name,
        #                                      class_mode="raw",
        #                                      target_size=(image_size, image_size),
        #                                      batch_size=12,
        #                                      shuffle=False)

        preds = model.predict(img_array, verbose=1)
        preds_result = np.argmax(preds, axis=1)

        save_path = ""
        if trueValue == "0" and preds_result[0] == 0:
            save_path = os.path.join(dstPath, "TN")
        elif trueValue == "0" and preds_result[0] == 1:
            save_path = os.path.join(dstPath, "FP")
        elif trueValue == "1" and preds_result[0] == 0:
            save_path = os.path.join(dstPath, "FN")
        elif trueValue == "1" and preds_result[0] == 1:
            save_path = os.path.join(dstPath, "TP")

        # following is heatmap generation
        fileNameWithoutExt, file_ext = os.path.splitext(fileName)
        originalFileName = fileNameWithoutExt + "_" + str(preds_result[0]) + "_" + str(trueValue) + ".jpg"
        heatMapFileName = fileNameWithoutExt + "_" + str(preds_result[0]) + "_" + str(trueValue) + "_heatmap.jpg"
        convFileName = fileNameWithoutExt + "_" + str(preds_result[0]) + "_" + str(trueValue) + "_convLayer.jpg"
        convLayerSavePath = os.path.join(save_path, convFileName)
        orginalFilePath = os.path.join(save_path, originalFileName)
        heatMapFIlePath = os.path.join(save_path, heatMapFileName)

        Path(save_path).mkdir(parents=True, exist_ok=True)

        heatmap = make_gradcam_heatmap(
            img_array, model, last_conv_layer_name, classifier_layer_names
        )
        # plt.matshow(heatmap)
        # heatmap_save = keras.preprocessing.image.array_to_img()
        # heatmap_save.save(convLayerSavePath)
        # plt.savefig(convLayerSavePath)
        # plt.close()

        # We load the original image
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)

        # original_img = keras.preprocessing.image.array_to_img(img)
        # original_img.save(orginalFilePath)

        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        heatmap = 255 - heatmap
        # # We use jet colormap to colorize heatmap
        # jet = cm.get_cmap("jet")
        #
        # # We use RGB values of the colormap
        # jet_colors = jet(np.arange(256))[:, :3]
        # jet_heatmap = jet_colors[heatmap]
        #
        # # We create an image with RGB colorized heatmap
        # jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        # jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        # superimposed_img = jet_heatmap * 0.4 + img
        # superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        image = cv2.imread(img_path)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        # heatmap[0] = 255 - heatmap[0]
        output = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        # output = heatmap * 0.4 + image

        cv2.rectangle(output, (0, 0), (450, 40), (0, 0, 0), -1)
        text_display = "true: " + str(trueValue) + " predicted: " + str(preds_result[0])
        cv2.putText(output, text_display, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        output = np.hstack([image, heatmap, output])
        superimposed_img = keras.preprocessing.image.array_to_img(output)
        superimposed_img.save(heatMapFIlePath)

        new_row = {'file_name': fileName, 'true_result': trueValue, 'predict_result': preds_result[0], 'confidence_score': preds[0]}
        exp_df = exp_df.append(new_row, ignore_index=True)

    return exp_df

heat_map_save_dir = os.path.join(processedHeatMapSavePath, "Val", last_conv_layer_name)

for index, row in validation.iterrows():
    print("Generating heatmap for image ", row["image_name"], " in layer top_activation for validation dataset")
    # try:
    export_df = generate_heat_map(row["image_name"], row["Class"], base_image_dir, heat_map_save_dir,export_df)
    # except:
    # print("An exception occurred when processing image ", row["image_name"])

export_save_dir = os.path.join(heat_map_save_dir, 'result.csv')
export_df.to_csv(export_save_dir)
export_df = pd.DataFrame([['', 0, 0, 0]], columns=['file_name', 'true_result', 'predict_result', 'confidence_score'])

heat_map_save_dir = os.path.join(processedHeatMapSavePath, "Test", last_conv_layer_name)

for index, row in test.iterrows():
    print("Generating heatmap for image ", row["image_name"], " in layer top_activation for test dataset")
    # try:
    export_df = generate_heat_map(row["image_name"], row["Class"], base_image_dir, heat_map_save_dir,export_df)
    # except:
    # print("An exception occurred when processing image ", row["image_name"])

export_save_dir = os.path.join(heat_map_save_dir, 'result.csv')
export_df.to_csv(export_save_dir)
export_df = pd.DataFrame([['', 0, 0, 0]], columns=['file_name', 'true_result', 'predict_result', 'confidence_score'])

last_conv_layer_name = "top_conv"

heat_map_save_dir = os.path.join(processedHeatMapSavePath, "Val", last_conv_layer_name)

# for index, row in validation.iterrows():
#     print("Generating heatmap for image ", row["SYS2"], row["image_name"], " in layer top_cov for validation dataset")
#     # try:
#     generate_heat_map(row["image_name"], base_image_dir, heat_map_save_dir)
#     # except:
#     #    print("An exception occurred when processing image ", row["image_name"])
#
# heat_map_save_dir = os.path.join(processedHeatMapSavePath, "Test", last_conv_layer_name)
#
# for index, row in test.iterrows():
#     print("Generating heatmap for image ", row["SYS2"], row["image_name"], " in layer top_cov for test dataset")
#     # try:
#     generate_heat_map(row["image_name"], base_image_dir, heat_map_save_dir)
#     # except:
#     #    print("An exception occurred when processing image ", row["image_name"])
