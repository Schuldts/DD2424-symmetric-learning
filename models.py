import os
import numpy as np
from keras.datasets import cifar10, fashion_mnist, mnist
from keras.utils import np_utils
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
import tensorflow as tf
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, LearningRateScheduler
import matplotlib.pyplot as plt
import argparse
import h5py
from skimage.transform import resize


def get_data(noise_ratio=0, dataset='cifar10'):
    num_classes = 10
    train_X, train_y, test_X, test_y = None, None, None, None
    if (dataset == 'cifar10'):
        (train_X, train_y), (test_X, test_y) = cifar10.load_data()

        train_X = train_X / 255.0
        test_X = test_X / 255.0

        mean = train_X.mean(axis=0)
        std = np.std(train_X)
        train_X = (train_X - mean) / std
        test_X = (test_X - mean) / std

        # they are 2D originally in cifar
        train_y = train_y.ravel()
        test_y = test_y.ravel()

    elif (dataset == 'fashion_mnist'):
        (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

        # Detta behövs för fashion minst, så att den också blir 4-dimensionell, annars är den 3-dimensionell
        train_X = train_X.reshape(-1, 28, 28, 1)
        test_X = test_X.reshape(-1, 28, 28, 1)

        train_X = train_X / 255
        test_X = test_X / 255

        train_y = train_y.copy()

        # May not be needed
        mean = train_X.mean(axis=0)
        std = np.std(train_X)
        train_X = (train_X - mean) / std
        test_X = (test_X - mean) / std
    
    elif (dataset == 'mnist'):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        # Detta behövs för fashion minst, så att den också blir 4-dimensionell, annars är den 3-dimensionell
        train_X = train_X.reshape(-1, 28, 28, 1)
        test_X = test_X.reshape(-1, 28, 28, 1)

        train_X = train_X / 255
        test_X = test_X / 255

        train_y = train_y.copy()

        # May not be needed
        mean = train_X.mean(axis=0)
        std = np.std(train_X)
        train_X = (train_X - mean) / std
        test_X = (test_X - mean) / std

    else: 
        print('Please enter either fashion_mnist, mnist or cifar10 as the dataset')

 
    train_y_clean = np.copy(train_y)

    if noise_ratio > 0: 
        # data_file = "%s_train_labels_%s.npy" % ("cifar-10", noise_ratio)
        
        # if os.path.isfile(data_file):
        #     train_y = np.load(data_file)

        # else:
        noisy = int(train_y.shape[0] * noise_ratio)
        indexes_per_class = [np.where(train_y_clean == i)[0] for i in range(num_classes)]
        noisy_per_class = int(noisy / num_classes)

        noisy_indexes = []
        for i in range(num_classes):
            new_noisy_indexes = np.random.choice(indexes_per_class[i], noisy_per_class, replace=False)
            noisy_indexes.extend(new_noisy_indexes)
        
        for i in noisy_indexes:
            old_class = train_y[i]
            new_class = list(range(num_classes))
            new_class.remove(old_class)
            # print("old_class ", old_class)
            # print("new_class ", new_class)
            train_y[i] = np.random.choice(new_class)
            # print("new_class ", train_y[i])
            #f = open(data_file, "x")
            #f.close()
            #np.save(data_file, train_y)
        
        # TODO: check for optimization 
        # np.save()
    
    # one-hot-encode the labels
    train_y_clean = np_utils.to_categorical(train_y_clean, num_classes)
    train_y = np_utils.to_categorical(train_y, num_classes)
    test_y = np_utils.to_categorical(test_y, num_classes)
    
    return train_X, train_y, test_X, test_y


def resize_usps(x):
    H, W, C = 28, 28, 1
    #x = x.reshape(16, 16, 1)
    resized_x = np.empty((len(x), H, W, C), dtype='float32')
    for i, img in enumerate(x):
        # resize returns [0, 1]
        resized_x[i] = resize(img, (H, W, C), mode='reflect')

    return resized_x


def get_usps():
    path = "data/usps.h5"
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        train_X = train.get('data')[:].reshape(-1, 16, 16, 1)
        train_y = train.get('target')[:]
        test = hf.get('test')
        test_X = test.get('data')[:].reshape(-1, 16, 16, 1)
        test_y = test.get('target')[:]
    
    train_X = resize_usps(train_X)
    test_X = resize_usps(test_X)
    return train_X, train_y, test_X, test_y



def get_model(input_shape, num_classes, dataset='cifar10'):
    input_layer = Input(shape=input_shape)
    if dataset == 'cifar10':
        # Block 1
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv1')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = Flatten(name='flatten')(x)

        x = Dense(256, kernel_initializer="he_normal", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation(tf.nn.softmax)(x)
    
    elif dataset == 'fashion_mnist' or dataset == 'mnist':
        # Block 1
        x = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", name='conv1')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        # Block 2
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        x = Flatten(name='flatten')(x)

        x = Dense(128, kernel_initializer="he_normal", name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)
        # they had a comment here
        # x = Dropout(0.2)(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation(tf.nn.softmax)(x)
    
    else:
        print('Please enter either fashion_mnist, mnist or cifar10 as the dataset')

    # Create model.
    model = Model(input_layer, x)
    return model


def cifar_scheduler(epoch):
    if epoch > 80:
        return 1e-4
    elif epoch > 40:
        return 1e-3
    else:
        return 1e-2

# Fashion mnist is an easier dataset so higher learning rate
def fashion_mnist_scheduler(epoch):
    if epoch > 30:
        return 1e-3
    elif epoch > 40:
        return 1e-2
    else:
        return 1e-1
    
class LoggerCallback(Callback):
    """
    Log train/val loss and acc into file for later plots.
    """
    def __init__(self, model, train_X, train_y, test_X, test_y, noise_ratio, epochs, alpha, beta):
        super(LoggerCallback, self).__init__()
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.n_class = train_y.shape[1]
        self.noise_ratio = noise_ratio
        # self.asym = asym
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta

        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.train_loss_class = [None]*self.n_class
        self.train_acc_class = [None]*self.n_class
        self.train_predictions = []
        self.valid_predictions = []

        # the followings are used to estimate LID
        self.lid_k = 20
        self.lid_subset = 128
        self.lids = []

        # complexity - Critical Sample Ratio (csr)
        self.csr_subset = 500
        self.csr_batchsize = 100
        self.csrs = []

    def on_epoch_end(self, epoch, logs={}):
        tr_acc = logs.get('acc')
        tr_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        # train_pred = self.model.predict(self.train_X)
        valid_pred = self.model.predict(self.test_X)

        self.train_loss.append(tr_loss)
        self.test_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.test_acc.append(val_acc)
        # self.train_predictions.append(train_pred)
        self.valid_predictions.append(valid_pred)

        print('ALL acc:', self.test_acc)

        # if self.asym:
        #     file_name = 'log/asym_loss_%s_%s.npy' % \
        #                 ("cifar-10", self.noise_ratio)
        #     np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss))))
        #     file_name = 'log/asym_acc_%s_%s.npy' % \
        #                 ("cifar-10", self.noise_ratio)
        #     np.save(file_name, np.stack((np.array(self.train_acc), np.array(self.test_acc))))
        #     file_name = 'log/asym_class_loss_%s_%s.npy' % \
        #                 ("cifar-10", self.noise_ratio)
        #     np.save(file_name, np.array(self.train_loss_class))
        #     file_name = 'log/asym_class_acc_%s_%s.npy' % \
        #                 ("cifar-10", self.noise_ratio)
        #     np.save(file_name, np.array(self.train_acc_class))
        
        file_name = 'log/loss_%s_%s_%s.npy' % \
                    ("cifar-10", self.noise_ratio, self.alpha)
        np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss))))
        file_name = 'log/acc_%s_%s_%s.npy' % \
                    ("cifar-10", self.noise_ratio, self.alpha)
        np.save(file_name, np.stack((np.array(self.train_acc), np.array(self.test_acc))))
class SGDLearningRateTracker(Callback):
    def __init__(self, model):
        super(SGDLearningRateTracker, self).__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        init_lr = float(backend.get_value(self.model.optimizer.lr))
        decay = float(backend.get_value(self.model.optimizer.decay))
        iterations = float(backend.get_value(self.model.optimizer.iterations))
        lr = init_lr * (1. / (1. + decay * iterations))
        print('init lr: %.4f, current lr: %.4f, decay: %.4f, iterations: %s' % (init_lr, lr, decay, iterations))
     
def acc_per_class(pred_val, true_val, n_epochs):
    predictions_val = []

    for j in range(n_epochs):
        predictions_val.append([np.argmax(i) for i in pred_val[j]])
    
    predictions_val = np.array(predictions_val)
    
    true_val = [np.where(true_val[i]==1)[0][0] for i in range(len(true_val))]

    accuracy_per_class_val = np.zeros((n_epochs, len(set(true_val))))
    

    for epoch in range(n_epochs):
        for i in set(true_val):
            index_per_class = np.where(true_val == i)[0]
            # print(predictions_val.shape)
            correct_guesses = np.where(predictions_val[epoch, index_per_class]==i)[0]
            # Define undefined as 1
            accuracy = 1
            if len(index_per_class) > 0:
                accuracy = len(correct_guesses) / len(index_per_class)
            
            accuracy_per_class_val[epoch, i] = accuracy
    
    return accuracy_per_class_val

def acc_per_class_usps(predictions, targets):
    acc_per_class = np.zeros((len(set(targets))))
    
    for i in set(targets):
        index_per_class = np.where(targets == i)[0]
        correct_guesses = np.where(predictions[index_per_class]==i)[0]
       
        accuracy = 1
        if len(index_per_class) > 0:
            accuracy = len(correct_guesses) / len(index_per_class)
        
        acc_per_class[i] = accuracy
    return acc_per_class

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)
        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.math.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.math.log(y_true_2), axis = -1))
    return loss

def train(n_batch, n_epochs, noise_ratio, type='ce', dataset=cifar10, alpha = 1.0, beta = 1.0, external_test=None):
    train_X, train_y, test_X, test_y = get_data(noise_ratio, dataset)
    n_images = train_X.shape[0]
    image_shape = train_X.shape[1:]
    num_classes = train_y.shape[1]

    model = get_model(image_shape, 10)
    loss = backend.categorical_crossentropy
        
    model.compile(
        loss = loss, 
        optimizer = SGD(learning_rate=0.1, decay=1e-4, momentum=0.9),
        metrics=['accuracy']
    )

    model_save_file = "model/%s_%s.{epoch:02d}.hdf5" % ("cifar-10", noise_ratio)

    # TODO: Maybe add callbacks/checkpoints
    callbacks = []
    cp_callback = ModelCheckpoint(model_save_file,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
        period=1)
    callbacks.append(cp_callback)

    if dataset == 'cifar10':
        scheduler = cifar_scheduler
    elif dataset == 'fashion_mnist' or dataset == 'mnist':
        scheduler = fashion_mnist_scheduler
    else:
        print("Please enter either fashion_mnist, cifar10, mnist")

    learning_rate_scheduler = LearningRateScheduler(scheduler)
    callbacks.append(learning_rate_scheduler)

    callbacks.append(SGDLearningRateTracker(model))

    log_callback = LoggerCallback(model, train_X, train_y, test_X, test_y, noise_ratio, n_epochs, alpha, beta)

    callbacks.append(log_callback)

    # TODO: Read up on this
    datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # Generate more data 
    datagen.fit(train_X)

    # main attraction
    history = model.fit_generator(datagen.flow(train_X, train_y, batch_size=n_batch),
    steps_per_epoch=train_X.shape[0] / n_batch, epochs=n_epochs,
    validation_data=(test_X, test_y),
    verbose=1,
    callbacks = callbacks,
    )
    if (external_test != None):
        external_test_y = external_test['y']
        external_test_x = external_test['x']
        predictions = model.predict(external_test_x)
        predicted_ys = np.argmax(predictions, axis=1)

        acc_external = acc_per_class_usps(predicted_ys, external_test_y)
        f = open(type+ "-" + dataset + "-" +str(noise_ratio)+" output.txt", "a")
        for i in range(acc_external.shape[0]):
            output_s = "class: " + str(i) + " final accuracy: " + str(acc_external[i])
            print(output_s)
            f.write(output_s)

        total_accuracy = np.sum([int(predicted_ys[i]==external_test_y[i]) for i in range(len(predicted_ys))])/len(predicted_ys)
        output_s = "total accuracy: " + str(total_accuracy)
        print(output_s)
        f.write(output_s)
        f.close()

    else:     
        acc_val = acc_per_class(log_callback.valid_predictions, test_y, n_epochs)
        for i in range(acc_val.shape[1]):
            print("class: " + str(i) + " final accuracy: " + str(acc_val[-1:, i]))
        print("final overall accuracy: " + str(history.history['val_accuracy'][-1]))
        plots = []
        #plt.plot(history.history['accuracy'], label='Accuracy training data')
        overall, = plt.plot(history.history['val_accuracy'],  color="black", label="Overall",)
        plots.append(overall)
        
        for i in range(acc_val.shape[1]):
            class_plot, = plt.plot(list(range(n_epochs)), acc_val[:, i], "--", label = "Class " + str(i))
            plots.append(class_plot)

        plt.legend(handles=plots)
        plt.title('Accuracy Overall and per Class')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(loc="lower right")
        if noise_ratio == 0:
            noise_ratio = "clean"
        plt.savefig(("plots/"+type+ "-" + dataset + "-" +str(noise_ratio)+".jpg"))
        plt.clf()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some stuff')
    parser.add_argument('dataset', type=str, help='the dataset to use, either fashion_mnist, mnist or cifar10')
    parser.add_argument('n_epochs', type=int, help='the number of epochs')
    parser.add_argument('n_batch', type=int, help='the batch size')
    parser.add_argument('use_external', type=int, help='if to use external dataset')

    args = parser.parse_args()

    dataset = args.dataset

    n_epochs =  args.n_epochs # 120

    n_batch = args.n_batch # 64

    external_test = None

    # if (args.use_external == 1):
    #     _, _, usps_X, usps_Y = get_usps()
    #     external_test = {'x': usps_X, 'y': usps_Y}

    # cross entropy with clean and 40% noisy data

    #train(n_batch, n_epochs, 0, "ce", dataset, external_test=external_test)

    train(n_batch, n_epochs, 0.4, "ce", dataset, external_test=external_test)

    # symmetric cross entropy with clean and 40% noisy data
    #train(n_batch, n_epochs, 0, "sce", dataset, external_test=external_test)

    #train(n_batch, n_epochs, 0.4,"sce", dataset, 0.1,1, external_test=external_test)
