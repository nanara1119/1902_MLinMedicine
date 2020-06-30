from keras import Input, Model
from keras import backend as K
from keras.layers import Conv1D, BatchNormalization, Activation, Dropout, TimeDistributed, Dense, MaxPooling1D, Lambda, Add
from keras.optimizers import Adam


def add_conv_weight(layer, filter_length, num_filters, subsample_length=1):
    layer = Conv1D(filters=num_filters, kernel_size=filter_length,
                   strides=subsample_length, padding='same', kernel_initializer="he_normal")(layer)
    return layer

def add_relu(layer, dropout = 0):
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)

    if dropout > 0 :
        layer = Dropout(0.2)(layer)

    return layer

def add_output_layer(layer):
    layer = TimeDistributed(Dense(4))(layer)
    return Activation("softmax")(layer)


def add_resnet_layers(layer):
    #   input block
    layer = add_conv_weight(layer = layer, filter_length=16, num_filters=32, subsample_length=1)
    layer = add_relu(layer)

    #   main loop block
    sub_sample = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    for i , sl in enumerate(sub_sample):
        num_filters = get_num_filters_at_index(i, 32)
        layer = resnet_block(layer, num_filters, sl, i)

    #   output block 의 BN / relu
    layer = add_relu(layer)
    return layer


def get_num_filters_at_index(index, num_start_filters):
    return 2**int(index / 4) * num_start_filters

def resnet_block(layer, num_filters, subsample_length, block_index):

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis = 2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length, padding ='same')(layer)
    zero_pad = (block_index % 4) == 0 and block_index > 0

    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape = zeropad_output_shape)(shortcut)

    for i in range(2) :
        if not(block_index == 0 and i == 0):
            layer = add_relu(layer, dropout=0.2 if i > 0 else 0)
        layer = add_conv_weight(layer, 16, num_filters, subsample_length if i == 0 else 1)
    layer = Add()([shortcut, layer])
    return layer

def first_block(inputs):
    layer = Conv1D(filters=32, kernel_size=16, strides=1, padding='same', kernel_initializer="he_normal")(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    shortcut = MaxPooling1D(pool_size=1, strides=1)(layer)
    layer = Conv1D(filters=32, kernel_size=16, strides=1, kernel_initializer="he_normal")(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv1D(filters=32, kernel_size=16, padding='same', strides=1, kernel_initializer='he_normal')(layer)
    return Add()([shortcut, layer])


def main_block(layer):
    filter_length = 32
    block_size = 15

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    for index in range(block_size):
        subsample_length = 2 ** int(index / 4) * 32
        shortcut = MaxPooling1D(pool_size=subsample_length, padding='same')(layer)

        if index % 4 == 0 and index > 0:
            shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
            filter_length *= 2

        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv1D(filters=filter_length,
                       kernel_size=16,
                       padding='same',
                       strides=subsample_length,
                       kernel_initializer='he_normal')(layer)

        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.2)(layer)
        layer = Conv1D(filters=filter_length,
                       kernel_size=16,
                       padding='same',
                       strides=1,
                       kernel_initializer='he_normal')(layer)
        layer = Add()([shortcut, layer])

    return layer


def output_block(layer):
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = TimeDistributed(Dense(4))(layer)
    layer = Activation('softmax')(layer)

    return layer


def add_compile(model):
    optimizer = Adam(lr=0.001, clipnorm=1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def build_model():
    inputs = Input(shape=[None, 1], dtype='float32', name='inputs')

    layer = add_resnet_layers(inputs)
    output = add_output_layer(layer)

    model = Model(inputs=inputs, output=output)
    add_compile(model)
    return model
