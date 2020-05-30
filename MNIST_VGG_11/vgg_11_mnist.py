# %% [code]
#import os, time, itertools, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow_datasets
import pickle
tf.compat.v1.disable_eager_execution()


import time
import os
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc
import pdb
import cv2
from keras.utils.np_utils import to_categorical



# %% [code]
def vgg_11_mnist(x,ndim):
    initializer = tf.initializers.GlorotUniform()

    weights = {
    'w1': tf.Variable(initializer(shape=(3,3,ndim,64)), name='w1'),
    'w2': tf.Variable(initializer(shape=(3, 3, 64, 128)), name='w2'),
    'w3': tf.Variable(initializer(shape=(3, 3, 128, 256)), name='w3'),
    'w4': tf.Variable(initializer(shape=(3, 3, 256, 256)), name='w4'),
    'w5': tf.Variable(initializer(shape=(3, 3, 256, 512)), name='w5'),
    'w6': tf.Variable(initializer(shape=(3, 3, 512, 512)), name='w6'),
    'w7': tf.Variable(initializer(shape=(3, 3, 512, 512)), name='w7'),
    'w8': tf.Variable(initializer(shape=(3, 3, 512, 512)), name='w8')
    }

    biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([128]), name='b2'),
      'b3': tf.Variable(tf.zeros([256]), name='b3'),
      'b4': tf.Variable(tf.zeros([256]), name='b4'),
      'b5': tf.Variable(tf.zeros([512]), name='b5'),
      'b6': tf.Variable(tf.zeros([512]), name='b6'),
      'b7': tf.Variable(tf.zeros([512]), name='b7'),
      'b8': tf.Variable(tf.zeros([512]), name='b8')
    }

    kernel = [2,2]

    ##conv1
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,weights['w1'],strides=[1,1,1,1],padding='SAME'),biases['b1']))
    print("conv1", conv1)
    maxconv1 = tf.nn.max_pool2d(conv1,ksize = kernel,strides = [1,2,2,1], padding='SAME')
    print(maxconv1)

    ##conv2
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxconv1,weights['w2'],strides=[1,1,1,1],padding='SAME'),biases['b2']))
    maxconv2 = tf.nn.max_pool2d(conv2,ksize = kernel,strides = [1,2,2,1], padding='SAME')
    print("conv2", conv2)
    print(maxconv2)
    ##conv3 and conv4
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxconv2,weights['w3'],strides=[1,1,1,1],padding='SAME'),biases['b3']))
    conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3,weights['w4'],strides=[1,1,1,1],padding='SAME'),biases['b4']))
    maxconv4 = tf.nn.max_pool2d(conv4,ksize = kernel,strides = [1,2,2,1], padding='SAME')
    print("conv3 and 4", conv3)
    print(conv4)
    print(maxconv4)
    ##conv5 and conv
    conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxconv4,weights['w5'],strides=[1,1,1,1],padding='SAME'),biases['b5']))
    conv6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5,weights['w6'],strides=[1,1,1,1],padding='SAME'),biases['b6']))
    maxconv6 = tf.nn.max_pool2d(conv6,ksize = kernel,strides = [1,2,2,1], padding='SAME')

    ##conv7 and conv8
    conv7 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxconv6,weights['w7'],strides=[1,1,1,1],padding='SAME'),biases['b7']))
    conv8 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv7,weights['w8'],strides=[1,1,1,1],padding='SAME'),biases['b8']))
    maxconv8 = tf.nn.max_pool2d(conv8,ksize = kernel,strides = [1,2,2,1], padding='SAME')
    print(maxconv8)

    inputfc = tf.compat.v1.layers.flatten(maxconv8)
    print(inputfc)

    ## Initializer
    w_init = tf.compat.v1.keras.initializers.glorot_uniform()
    b_init = tf.constant_initializer(0.)

    ## 9
    w9 = tf.compat.v1.get_variable('w9', [inputfc.get_shape()[1],4096], initializer=w_init)
    b9 = tf.compat.v1.get_variable('b9', [4096], initializer=b_init)
    h9 = tf.nn.dropout(tf.nn.relu(tf.matmul(inputfc, w9) + b9), rate=0.5)

    ##10
    w10 = tf.compat.v1.get_variable('w10', [w9.get_shape()[1], 4096], initializer=w_init)
    b10 = tf.compat.v1.get_variable('b10', [4096], initializer=b_init)
    h10 = tf.nn.dropout(tf.nn.relu(tf.matmul(h9, w10) + b10), rate=0.5)

    ##11
    w11 = tf.compat.v1.get_variable('w11', [w10.get_shape()[1], 1000], initializer=w_init)
    b11 = tf.compat.v1.get_variable('b11', [1000], initializer=b_init)
    h11 = tf.nn.dropout(tf.nn.relu(tf.matmul(h10, w11) + b11), rate=0.0)


    w12 = tf.compat.v1.get_variable('w12', [w11.get_shape()[1], 10], initializer=w_init)
    b12 = tf.compat.v1.get_variable('b12', [10], initializer=b_init)
    h12 = tf.matmul(h11, w12) + b12
    h122 = tf.nn.softmax(h12)

   # prediction = tf.argmax(input=h122,axis=1)
   # print("prediction is",prediction)

    return h122

# %% [code]
 

def resizeimg(x):
    arr = []
    for img in x:
        newimg = list(cv2.resize(img,dsize=(50,50), interpolation=cv2.INTER_CUBIC))
        arr.append(newimg)
    nparrimg = np.array(arr)
    nparrimg = np.expand_dims(nparrimg, axis=-1)
    return nparrimg



def mnistloaddata():
    mnist = tf.keras.datasets.mnist
    print(mnist)
    (x_train, y_train), (x_test,y_test) = mnist.load_data()
    x_train = (x_train - 0.5) / 0.5
    x_test = (x_test - 0.5) / 0.5
    
    x_test = resizeimg(x_test)
    
    # ##getvalidationdata
    valshape = x_test.shape[0]
    x_train = x_train[valshape:]
    y_train = y_train[valshape:]

    x_validate = x_train[:valshape]
    y_validate = y_train[:valshape]
    
    return x_train, y_train, x_test, y_test, x_validate, y_validate


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    
    x = range(len(hist['traininglosses']))
    
    fig, (vax, hax) = plt.subplots(2,1)
    #ax = ax.ravel()
    l = [1,2]
    
    y1 = hist['traininglosses']
    y2 = hist['validationlosses']
    
    y3 = hist['trainingaccuracies']
    y4 = hist['validationaccuracies']
    
    vax.plot(x, y1, label='trainingloss')
    vax.plot(x, y2, label='validationlosses')

    vax.set_xlabel('Epoch')
    vax.set_ylabel('Loss')
    
    vax.legend(loc=1)
    vax.grid(True)
    
    hax.plot(x, y3, label='training_acc')
    hax.plot(x, y4, label='validation_acc')
    
    hax.set_xlabel('Epoch')
    hax.set_ylabel('Accuracy')
    
    hax.legend(loc=4)
    hax.grid(True)
    
    
    fig.suptitle("MNIST DATA WITH NORMAL VGG-11 ARCHITECTURE")

    



    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
        
      



# %% [code]
# loaw MNIST
x_train, y_train, x_test, y_test,x_validate, y_validate = mnistloaddata()
dimensionx = x_train.shape[1]
print(dimensionx)
dimensiony = x_train.shape[2]
print(dimensiony)

y_test = to_categorical(y_test,10)


# print(x_validate.shape)

# training parameters
batch_size = 150
lr = 0.0001
train_epoch = 12
input_x = tf.compat.v1.placeholder(dtype=tf.float32,name='Placeholder1')
input_y = tf.compat.v1.placeholder(dtype=tf.float32,name='Placeholder2')


with tf.compat.v1.variable_scope('D') as scope:
    cfa_x = tf.compat.v1.placeholder(tf.float32, shape=(None,50,50,1),name='Placeholderx')
    Cfa = vgg_11_mnist(cfa_x,1)
    scope.reuse_variables()

#loss_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y,logits=Cfa))
loss_f = tf.reduce_mean(-tf.math.reduce_sum(input_y * tf.math.log(Cfa), axis=[1]))
weight_decay = 5e-4
reg_loss = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
#l2_loss = weight_decay * tf.math.add_n([tf.nn.l2_loss(tf.cast(v,tf.float32)) for v in tf.trainable_variables()])
loss_f = loss_f + (weight_decay * sum(reg_loss))


print(loss_f)
opt = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss_f)
correct_prediction = tf.equal(tf.argmax(Cfa,1), tf.argmax(input_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.compat.v1.InteractiveSession()
tf.compat.v1.global_variables_initializer().run()

# results save folder
if not os.path.isdir('MNIST_VGG11_results'):
    os.mkdir('MNIST_VGG11_results')
if not os.path.isdir('MNIST_VGG11_results/results'):
    os.mkdir('MNIST_VGG11_results/results')
train_hist = {}
train_hist['traininglosses'] = []
train_hist['validationlosses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['trainingaccuracies'] = []
train_hist['validationaccuracies'] = []
train_hist['total_ptime'] =[]
# training-loop
np.random.seed(int(time.time()))
start_time = time.time()

top_c, _ = tf.math.in_top_k()

saver = tf.compat.v1.train.Saver()

for epoch in range(train_epoch):
    losses = []
    taccuracies = []
    valaccuracies = []
    vallosses = []
    testaccuracies = []
    epoch_start_time = time.time()
    for iter in range(x_train.shape[0] // batch_size):
        x_ = x_train[iter*batch_size:(iter+1)*batch_size]
        y_ = y_train[iter*batch_size:(iter+1)*batch_size]
        
        x_ = resizeimg(x_)

        y_ = to_categorical(y_, num_classes=10)
        
        loss_d_, _,acc_t  = sess.run([loss_f, opt, accuracy], feed_dict={cfa_x: x_, input_y:y_})
        losses.append(loss_d_)
        taccuracies.append(acc_t)
        
    for iter in range(x_validate.shape[0] // batch_size):
        x_val = x_validate[iter*batch_size:(iter+1)*batch_size]
        y_val = y_validate[iter*batch_size:(iter+1)*batch_size]
        
        x_val = resizeimg(x_val)
        y_val = to_categorical(y_val, num_classes=10)
        
        acc_v, loss_val = sess.run([accuracy, loss_f], feed_dict={cfa_x: x_val, input_y:y_val})
        valaccuracies.append(acc_v)
        vallosses.append(loss_val)
        
    if epoch >= 10:
        for iter in range(x_test.shape[0] // batch_size):
            x_tt = x_test[iter*batch_size:(iter+1)*batch_size]
            y_tt = y_test[iter*batch_size:(iter+1)*batch_size]
            
            acctest = accuracy.eval({cfa_x: x_tt, input_y: y_tt})
            #print("Accuracy:", acctest)
            testaccuracies.append(acctest)
            
        print("Average Test Accuracy", np.mean(testaccuracies))
        
            
        
            
            
        saver.save(sess, 'checkpoints/skip-gram', global_step=epoch)
        
    
    
   # acc = sess.run(accuracy, feed_dict={cfa_x: x_res, input_y:y_)
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f t_acc: %.3f t_loss: %.3f val_acc: %.3f val_loss: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(taccuracies), np.mean(losses), np.mean(valaccuracies), np.mean(vallosses)))
    
    p = 'MNIST_VGG11_results/results/MNIST_VGG11_' + str(epoch + 1) + '.png'
    #show_result((epoch + 1), save=True, path=p)
        
    train_hist['traininglosses'].append(np.mean(losses))
    train_hist['validationlosses'].append(np.mean(vallosses))
    train_hist['trainingaccuracies'].append(np.mean(taccuracies))
    train_hist['validationaccuracies'].append(np.mean(valaccuracies))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()

total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")


    
    
    

    

# %% [code]
with open('MNIST_VGG11_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
    show_train_hist(train_hist, save=True, path='MNIST_VGG11_results/MNIST_VGG11_train_hist.png')
    images = []
    sess.close()



pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
gt = np.argmax(y_test,axis=1)
misclass = np.where(gt != pred)[0]
missed = misclass[:10]

labels = ['0','1','2','3','4','5','6','7','8','9','10']

fig, ax = plt.subplots(2,5)
fig.tight_layout()
ix = 0

for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_test[missed[ix],:,:,0], cmap='gray')
        ax[i,j].axis('off')
        ax[i,j].set_title('Predicted' +labels[pred[missed[ix]]]+ '\n Actual' + labels[gt[missed[ix]]])
        ix +=1
        
p = 'results/Predictions' + '.png'

plt.savefig(p)

plt.show()

prediction = tf.argmax(Cfa,1)
pred = sess.run(prediction, feed_dict=({cfa_x: x_test, input_y: y_test}))
gt = np.argmax(y_test,axis=1)
misclass = np.where(gt != pred)[0]
missed = misclass[:10]
print(missed)

labels = ['0','1','2','3','4','5','6','7','8','9','10']

fig, ax = plt.subplots(2,5)
fig.tight_layout()
ix = 0

for i in range(2):
    for j in range(5):
        ax[i,j].imshow(desiredtest[missed[ix],:,:,0], cmap='gray')
        ax[i,j].axis('off')
        ax[i,j].set_title('Predicted' +labels[pred[missed[ix]]]+ '\n Actual' + labels[gt[missed[ix]]])
        ix +=1
        
p = 'MNIST_VGG11_results/results/Predictions' + '.png'

plt.savefig(p)

plt.show()

# %% [code]
with open('MNIST_VGG11_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
    show_train_hist(train_hist, save=True, path='MNIST_VGG11_results/MNIST_VGG11_train_hist.png')
    images = []
    sess.close()

