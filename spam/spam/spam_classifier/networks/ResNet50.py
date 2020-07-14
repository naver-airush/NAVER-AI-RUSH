from tensorflow.keras import Input, Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Flatten, Dense
from os import path


def frozen_resnet(input_size, n_classes, local_weights="/resnets/resnet50v2_notop.h5"):
    if local_weights and path.exists(local_weights):
        print(f'Using {local_weights} as local weights.')
        model_ = ResNet50V2(
            include_top=False,
            input_tensor=Input(shape=input_size),
            weights=local_weights)
    else:
        print(
            f'Could not find local weights {local_weights} for ResNet. Using remote weights.')
        model_ = ResNet50V2(
            include_top=False,
            input_tensor=Input(shape=input_size))
    for layer in model_.layers:
        layer.trainable = False
    x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
    frozen_model = Model(model_.input, x)

    return frozen_model
