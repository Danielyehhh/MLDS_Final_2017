
# coding: utf-8

# In[2]:


import keras
from keras.layers import Add,Input,Dense,Activation,Conv2DTranspose,Reshape,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential, load_model
from keras import backend as K


# In[7]:


inputs = Input(shape = (240,320,3))

Conv_1_1 = Conv2D(filters = 32,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(inputs)
Conv_1_1 = LeakyReLU()(Conv_1_1)
Conv_1_2 = Conv2D(filters = 32,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Conv_1_1)
Conv_1_2 = LeakyReLU()(Conv_1_2)
Conv_1 = Add()([Conv_1_1,Conv_1_2])
Conv_1 = Conv2D(filters = 32,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Conv_1)
Conv_1 = LeakyReLU()(Conv_1)
Pool_1 = MaxPooling2D((2,2))(Conv_1)

Conv_2_1 = Conv2D(filters = 64,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Pool_1)
Conv_2_1 = LeakyReLU()(Conv_2_1)
Conv_2_2 = Conv2D(filters = 64,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Conv_2_1)
Conv_2_2 = LeakyReLU()(Conv_2_2)
Conv_2 = Add()([Conv_2_1,Conv_2_2])
Conv_2 = Conv2D(filters = 64,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Conv_2)
Conv_2 = LeakyReLU()(Conv_2)
Pool_2 = MaxPooling2D((2,2))(Conv_2)

Conv_3_1 = Conv2D(filters = 128,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Pool_2)
Conv_3_1 = LeakyReLU()(Conv_3_1)
Conv_3_2 = Conv2D(filters = 128,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Conv_3_1)
Conv_3_2 = LeakyReLU()(Conv_3_2)
Conv_3 = Add()([Conv_3_1,Conv_3_2])
Conv_3 = Conv2D(filters = 128,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Conv_3)
Conv_3 = LeakyReLU()(Conv_3)
Pool_3 = MaxPooling2D((2,2))(Conv_3)

Conv_4_1 = Conv2D(filters = 256,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Pool_3)
Conv_4_1 = LeakyReLU()(Conv_4_1)
Conv_4_2 = Conv2D(filters = 256,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Conv_4_1)
Conv_4_2 = LeakyReLU()(Conv_4_2)
Conv_4 = Add()([Conv_4_1,Conv_4_2])
Conv_4 = Conv2D(filters = 256,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Conv_4)
Conv_4 = LeakyReLU()(Conv_4)
Pool_4 = MaxPooling2D((2,2))(Conv_4)


Encoder = Conv2D(filters = 1024,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(Pool_4)
Encoder = LeakyReLU()(Encoder)
Encoder = Conv2D(filters = 1024,
                  kernel_size = 1,
                  strides = 1,
                  padding = 'same')(Encoder)
Encoder = LeakyReLU()(Encoder)

deconv_1 = Conv2DTranspose(filters = 128,
                           kernel_size = 3,
                           strides = 2,
                           padding = 'same')(Encoder)
deconv_1 = LeakyReLU()(deconv_1)
merge_1 = Add()([Pool_3,deconv_1])
merge_1 = Conv2D(filters = 128,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(merge_1)
merge_1 = LeakyReLU()(merge_1)

deconv_2 = Conv2DTranspose(filters = 64,
                           kernel_size = 3,
                           strides = 2,
                           padding = 'same')(merge_1)
deconv_2 = LeakyReLU()(deconv_2)
merge_2 = Add()([Pool_2,deconv_2])
merge_2 = Conv2D(filters = 64,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same')(merge_2)
merge_2 = LeakyReLU()(merge_2)

Decoder = Conv2DTranspose(filters = 64,
                           kernel_size = 3,
                           strides = 4,
                           padding = 'same')(merge_2)
Decoder = LeakyReLU()(Decoder)
Decoder = Conv2D(filters = 3,
                  kernel_size = 3,
                  strides = 1,
                  padding = 'same',
                  activation = 'softmax')(Decoder)


FCN_model = Model(inputs = inputs,
                  outputs = Decoder)
FCN_model.summary()

FCN_model.save('FCN_model.h5')

