from database import RestaurantDB
import io
import numpy as np
from PIL import Image
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py

def convert(row):
    return np.array(Image.open(io.BytesIO(row[0]))), row[1]

db = RestaurantDB() # connection to restaurant database

batch = db.get_nn_training_set(limit=10000, offset=0) # get one training batch of size 10000 starting at top of records
batch = map(convert, batch)
batch = np.asarray(batch)
x = batch[:,0]
y = batch[:,1]
labels = [[i,list(y).count(i)] for i in set(y)]


cnt0 = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0
Ytrain = []
# Iterate through our batch, save the images to a folder by label name for traing
for i in range(8400):
    im = Image.fromarray(x[i])
    if y[i] == 'Fine Dining' and cnt0 < 1000:
        cnt0 +=1
        idx = 3
        Ytrain.append([0,1,0])
        im.save("data/fine_dining/image_" + str(i) +".jpeg")
    elif y[i] == 'Casual Dining' and cnt1< 1000:
        cnt1+=1
        idx = 0
        Ytrain.append([1,0,0])
        im.save("data/casual_dining/image_" + str(i) +".jpeg")        
    elif y[i] == 'Casual Elegant' and cnt3 < 1000:
        cnt3+=1
        im.save("data/casual_elegant/image_" + str(i) +".jpeg")
        idx = 2
        Ytrain.append([0,0,1])
Ytrain = np.asarray(Ytrain)

cnt0 = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0
Yval = []
# Iterate through our batch, save the images to a folder by label name for validation
for i in range(8400,10000):
    im = Image.fromarray(x[i])
    if y[i] == 'Fine Dining' and cnt0 < 100:
        cnt0 +=1
        idx = 3
        Yval.append([0,1,0])
        im.save("validation/fine_dining/image_" + str(i) +".jpeg")
    elif y[i] == 'Casual Dining' and cnt1< 100:
        cnt1+=1
        idx = 0
        Yval.append([1,0,0])
        im.save("validation/casual_dining/image_" + str(i) +".jpeg")        
    elif y[i] == 'Casual Elegant' and cnt3 < 100:
        cnt3+=1
        im.save("validation/casual_elegant/image_" + str(i) +".jpeg")
        idx = 2
        Yval.append([0,0,1])
Yval = np.asarray(Yval)



# path to the model weights file.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images
img_width, img_height = 224, 224

# Setting up trainign and validation set
train_data_dir = 'data'
validation_data_dir = 'validation'
nb_train_samples = 3000
nb_validation_samples = 300
nb_epoch = 2


# Setting up an image generator which would apply random zoomin,
# rotation oand flipping
datagen = ImageDataGenerator(dim_ordering='th',
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            fill_mode='nearest')
                            
f = h5py.File(weights_path)
# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3,img_width, img_height), dim_ordering='th'))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1),dim_ordering='th'))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', dim_ordering='th'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', dim_ordering='th'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', dim_ordering='th'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', dim_ordering='th'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))

model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2', dim_ordering='th'))
model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3', dim_ordering='th'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))
print model.output_shape

# load the weights of the VGG16 networks

assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# Turn of trainability since they are pretty well trained already
for layer in model.layers:
    layer.trainable = False

# creating the fully connect head
topmodel = Sequential()
topmodel.add(Flatten(input_shape=model.output_shape[1:]))
topmodel.add(Dense(256, activation='relu'))
topmodel.add(Dropout(0.5))
topmodel.add(Dense(256, activation='relu'))
topmodel.add(Dropout(0.5))
topmodel.add(Dense(3, activation='softmax'))


# add the model on top of the convolutional base
model.add(topmodel)


# Setting up training data generator
generatorTrain = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=20,
        class_mode='categorical',
        shuffle=True)


# Setting up validation data generator
generatorVal = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=20,
        class_mode='categorical',
        shuffle=True)

# Setting up optimizer
opt = keras.optimizers.SGD(lr=0.01)

# Compile our model    
model.compile(optimizer='opt', loss='categorical_crossentropy', metrics=['accuracy'])

 
# Train our model
history = model.fit_generator(generatorTrain,
          samples_per_epoch = 500,
          nb_epoch=nb_epoch,
          validation_data=generatorVal,
          verbose =1,
          nb_val_samples = 100)

# Save the weights of the model          
model.save_weights(top_model_weights_path)
print 'Training done'