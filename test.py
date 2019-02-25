from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
(xtrain,ytrain),(xtest,ytest) = cifar10.load_data()
print('xtrain.shape',xtrain.shape)
print('ytrain.shape',ytrain.shape)
print('ytest.shape',ytest.shape)
print('xtest.shape',xtest.shape)
batchsize=200
cats = 10
nepoch = 100
xtrain = xtrain.reshape(50000,3072)

xtest = xtest.reshape(10000,3072)
xtrain = xtrain/255
xtest = xtest/255
ytrain = np_utils.to_categorical(ytrain,cats)
ytest = np_utils.to_categorical(ytest,cats)

model = Sequential()
model.add(Dense(units=10,input_shape=(3072,),activation='softmax',kernel_initializer='normal'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.summary()
history = model.fit(xtrain, ytrain, nb_epoch=nepoch, batch_size=batchsize, verbose=1)

# Evaluate
evaluation = model.evaluate(xtest, ytest, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))