#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import warnings
import pickle
#from scipy.speical import expit
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.special import expit as sigmoid

import tensorflow as tf
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
import cv2
import sys
import tensorflow.keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.callbacks import ModelCheckpoint


WIDTH_NORM = 224
HEIGHT_NORM = 224
GRID_NUM = 11
X_SPAN = WIDTH_NORM / GRID_NUM
Y_SPAN = HEIGHT_NORM / GRID_NUM
X_NORM = WIDTH_NORM / GRID_NUM
Y_NORM = HEIGHT_NORM / GRID_NUM


# In[4]:


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = conv2D(filters1, (1, 1), name = conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '2b')(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + "2b")(x)
    x = Activation('relu')(x)
    
    x = conv2D(filters3, (1, 1), name = conv_name_base + '2c')(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2c')(x)
    
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    
    return x
        


# In[5]:


def conv_block(input_tensor, kernel_size, filters, stage, block, strides = (2, 2)):
    filters1, filters2, filters3 = filters
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters1, (1, 1), strides = strides, name = conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '2b')(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters3, (1, 1), name = conv_name_base + '2c')(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2c')(x)
    
    shortcut = Conv2D(filters3, (1, 1), strides = strides, name = conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis = bn_axis, name = bn_name_base + '1')(shortcut)
    
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    
    return x


# In[6]:


def ResNet50(include_top = False, load_weight = True, weights = 'imagenet', input_tensor = None, input_shape = None, pooling = None, classes = 1000):
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('pass')
    
    input_shape = _obtain_input_shape(input_shape, default_size = 224, min_size = 197, data_format = K.image_data_format(), require_flatten = include_top)
    
    if input_tensor is None:
        img_input = Input(shape = input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(x)
    x = BatchNormalization(axis = bn_axis, name = 'bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides = (2, 2))(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage = 2, block = 'a', strides = (1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage = 2, block = 'b')
    x = identity_block(x, 3, [64, 64, 256], stage = 2, block = 'c')
    
    x = conv_block(x, 3, [128, 128, 512], stage = 3, block = 'a')
    x = identity_block(x, 3, [128, 128, 512], stage = 3, block = 'b')
    x = identity_block(x, 3, [128, 128, 512], stage = 3, block = 'c')
    x = identity_block(x, 3, [128, 128, 512], stage = 3, block = 'd')
    
    x = conv_block(x, 3, [256, 256, 1024], stage = 4, block = 'a')
    x = identity_block(x, 3, [256, 256, 1024], stage = 4, block = 'b')
    x = identity_block(x, 3, [256, 256, 1024], stage = 4, block = 'c')
    x = identity_block(x, 3, [256, 256, 1024], stage = 4, block = 'd')
    x = identity_block(x, 3, [256, 256, 1024], stage = 4, block = 'e')
    x = identity_block(x, 3, [256, 256, 1024], stage = 4, block = 'f')
    
    x = conv_block(x, 3, [512, 512, 2048], stage = 5, block = 'a')
    x = identity_block(x, 3, [512, 512, 2048], stage = 5, block = 'b')
    x = identity_block(x, 3, [512, 512, 2048], stage = 5, block = 'c')
    
    x = AveragePooling2D((7, 7), name = 'avg_pool')(x)
    
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation = 'softmax', name = 'fc1000')(x)
    else:
        x = Flatten(name = 'yolo_clf_0')(x)
        x = Dense(2048, activation = 'relu', name = 'yolo_clf_1')(x)
        x = Dropout(0.5, name = 'yolo_clf_2')(x)
        
        x = Dense(11 * 11 * (3 + 2 * 5), activation = 'linear', name = 'yolo_clf_3')(x)
        
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = Model(inputs, x, name = 'resnet50_yolo')
    
    if load_weight:
        if weights == 'imagenet':
            if include_top:
                weights_path = 'models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
            else:
                weights_path = 'models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        else:
            weights_path = weights
        print(weights_path, '\n', save_prefix, '\n', learning_rate) 
        
        model.load_weights(weights_path, by_name = True)
        
    return model


# In[7]:


def loop_body(t_true, t_pred, i, ta):
    c_true = t_true[i]
    c_pred = t_pred[i]
    
    # 텐서 좌표에 sigmoid 도포하여 예상대로 0~1 사이 스케일링
    c_pred = tf.concat((c_pred[:605], tf.sigmoid(c_pred[-968:])), axis = 0)
    
    xywh_true = tf.reshape(c_true[-968:], (11, 11, 2, 4))
    xywh_pred = tf.reshape(c_pred[-968:], (11, 11, 2, 4))
    
    # 정규화된 값을 실제 값으로 변환(그리드 셀 크기에 비례)
    x_true = xywh_true[:, :, :, 0] * X_NORM
    x_pred = xywh_pred[:, :, :, 0] * X_NORM
    
    y_true = xywh_true[:, :, :, 1] * Y_NORM
    y_pred = xywh_pred[:, :, :, 1] * Y_NORM
    
    w_true = xywh_true[:, :, :, 2] * WIDTH_NORM
    w_pred = xywh_pred[:, :, :, 2] * WIDTH_NORM
    
    h_true = xywh_true[:, :, :, 3] * HEIGHT_NORM
    h_pred = xywh_pred[:, :, :, 3] * HEIGHT_NORM
    
    x_dist = tf.abs(tf.subtract(x_true, x_pred)) # tensorflow 뺄셈
    y_dist = tf.abs(tf.subtract(y_true, y_pred))
    
    wwd = tf.nn.relu(w_true / 2 + w_pred / 2 - x_dist)
    hhd = tf.nn.relu(h_true / 2 + h_pred / 2 - y_dist)
    
    area_true = tf.multiply(w_true, h_true) # tensorflow 곱셈
    area_pred = tf.multiply(w_pred, h_pred)
    area_intersection = tf.multiply(wwd, hhd)
    
    # iou : 보통 두 가지 물체의 위치(Bounding Box)가 얼마나 일치하는지를 수학적으로 나타내는 지표
    iou = area_intersection / (area_true + area_pred - area_intersection + 1e-4)
    confidence_true = tf.reshape(iou, (-1, ))
    
    grid_true = tf.reshape(c_true[:363], (11, 11, 3))
    grid_true_sum = tf.reduce_sum(grid_true, axis = 2)
    grid_true_exp = tf.stack((grid_true_sum, grid_true_sum), axis = 2)
    grid_true_exp3 = tf.stack((grid_true_sum, grid_true_sum, grid_true_sum), axis = 2)
    grid_true_exp4 = tf.stack((grid_true_sum, grid_true_sum, grid_true_sum, grid_true_sum), axis = 2)
    
    coord_mask = tf.reshape(grid_true_exp4, (-1, ))
    confidence_mask = tf.reshape(grid_true_exp, (-1, ))
    confidence_true = confidence_true * confidence_mask
    
    # 계산된 신뢰도 값에 기반하고 비객체 그리드가 억제된 수정된 지상 실측 텐서
    c_true_new = tf.concat([c_true[:363], confidence_true, c_true[-968:]], axis = 0)
    
    # 손실 계산을 위해 그리드 셀에 '책임 있는' 경계 상자에 대한 마스크 생성
    confidence_true_matrix = tf.reshape(confidence_true, (11, 11, 2))
    confidence_true_argmax = tf.argmax(confidence_true_matrix, axis = 2)
    confidence_true_argmax = tf.cast(confidence_true_argmax, tf.int32)
    ind_i, ind_j = tf.meshgrid(tf.range(11), tf.range(11), indexing = 'ij') # 좌표벡터를 사용해 행렬을 만듬
    ind_argmax = tf.stack((ind_i, ind_j, confidence_true_argmax), axis = 2)
    ind_argmax = tf.reshape(ind_argmax, (121, 3))
    
    responsible_mask_2 = tf.scatter_nd(ind_argmax, tf.ones((121)), [11, 11, 2])
    responsible_mask_2 = tf.reshape(responsible_mask_2, (-1, ))
    responsible_mask_2 = responsible_mask_2 * confidence_mask
    
    responsible_mask_4 = tf.scatter_nd(ind_argmax, tf.ones((121, 2)), [11, 11, 2, 2])
    responsible_mask_4 = tf.reshape(responsible_mask_4, (-1, ))
    responsible_mask_4 = responsible_mask_4 * coord_mask
    
    # 나머지 경계 상자에 대한 mask
    inv_responsible_mask_2 = tf.cast(tf.logical_not(tf.cast(responsible_mask_2, tf.bool)), tf.float32)
    inv_responsible_mask_4 = tf.cast(tf.logical_not(tf.cast(responsible_mask_4, tf.bool)), tf.float32)
    
    # lambda 값 정의
    lambda_coord = 5.0
    lambda_noobj = 0.5
    
    # loss dimensions
    dims_true = tf.reshape(c_true_new[-968:], (11, 11, 2, 4))
    dims_pred = tf.reshape(c_pred[-968:], (11, 11, 2, 4))
    
    xy_true = tf.reshape(dims_true[:, :, :, :2], (-1, ))
    xy_pred = tf.reshape(dims_pred[:, :, :, :2], (-1, ))
    
    wh_true = tf.reshape(dims_true[:, :, :, 2:], (-1, ))
    wh_pred = tf.reshape(dims_pred[:, :, :, 2:], (-1, ))
    
    # xy difference loss
    xy_loss = (xy_true - xy_pred) * responsible_mask_4
    xy_loss = tf.square(xy_loss)
    xy_loss = lambda_coord * tf.reduce_sum(xy_loss)
    
    # wh 제곱근 difference loss
    wh_loss = (tf.sqrt(wh_true) - tf.sqrt(tf.abs(wh_pred))) * responsible_mask_4
    wh_loss = tf.square(wh_loss)
    wh_loss = lambda_coord * tf.reduce_sum(xy_loss)
    
    # conf losses
    conf_true = c_true_new[363:605]
    conf_pred = c_pred[363:605]
    
    conf_loss_obj = (conf_true - conf_pred) * responsible_mask_2
    conf_loss_obj = tf.square(conf_loss_obj)
    conf_loss_obj = tf.reduce_sum(conf_loss_obj)
    
    conf_loss_noobj = (conf_true - conf_pred) * inv_responsible_mask_2
    conf_loss_noobj = tf.square(conf_loss_noobj)
    conf_loss_noobj = lambda_noobj * tf.reduce_sum(conf_loss_noobj)
    
    # class prediction loss
    class_true = tf.reshape(c_true_new[:363], (11, 11, 3))
    class_pred = tf.reshape(c_pred[:363], (11, 11, 3))
    class_pred_softmax = class_pred # tf.nn.softmax(class_pred)
    
    classification_loss = class_true - class_pred_sotfmax
    classification_loss = classification_loss * grid_true_exp3
    classification_loss = tf.square(classification_loss)
    classification_loss = tf.reduce_sum(classification_loss)
    
    total_loss = xy_loss + wh_loss + conf_loss_obj + conf_loss_noobj + classification_loss
    
    ta = ta.write(i, total_loss)
    i = i + 1
    
    return t_true, t_pred, i, ta
    
    
    
    
    


# In[8]:


def custom_loss(y_true, y_pred):
    # 기본값이 없기 때문에 keras에서 손실 함수를 구함
    c = lambda t, p, i, ta : tf.less(i, tf.shape(t)[0])
    ta = tf.TensorArray(tf.float32, size = 1, dynamic_size = True)
    
    t, p, i, ta = tf.while_loop(c, loop_body, [y_true, y_pred, 0, ta])
    
    loss_tensor = ta.stack()
    loss_mean = tf.reduce_mean(loss_tensor)
    
    return loss_mean


# In[9]:


def coord_translate(bboxes, tr_x, tr_y):
    new_list = []
    for box in bboxes:
        coords = np.array(box[0])
        coords[:, 0] = coords[:, 0] + tr_x
        coords[:, 1] = coords[:, 1] + tr_y
        coords = coords.astype(np.int64)
        out_of_bound_indices = np.average(coords, axis = 0) >= 224
        
        if out_of_bound_indices.any():
            continue
        coords = coords.tolist()
        new_list.append((coords, box[1]))
    return new_list


# In[10]:


def coord_scale(bboxes, sc):
    new_list = []
    for box in bboxes:
        coords = np.array(box[0])
        coords = coords * sc
        coords = coords.astype(np.int64)
        out_of_bound_indices = np.average(coords, axis = 0) >= 224
        if out_of_bound_indices.any():
            continue
        coords = coordsd.tolist()
        new_list.append((coords, box[1]))
    return new_list


# In[11]:


def label_to_tensor(frame, imgsize = (224, 224), gridsize = (11, 11), classes = 3, bboxes = 2):
    grid = np.zeros(gridsize)
    
    y_span = imgsize[0] / gridsize[0]
    x_span = imgsize[1] / gridsize[1]
    
    class_prob = np.zeros((gridsize[0], gridsize[1], classes))
    confidence = np.zeros((gridsize[0], gridsize[1], bboxes))
    dims = np.zeros((gridsize[0], gridsize[1], bboxes, 4))
    
    for box in frame:
        ((x1, y1), (x2, y2)), (c1, c2, c3) = box
        x_grid = int(((x1 + x2) / 2) // x_span)
        y_grid = int(((y1 + y2) / 2) // y_span)
        
        class_prob[y_grid, x_grid] = (c1, c2, c3)
        
        x_center = ((x1 + x2) / 2)
        y_center = ((y1 + y2) / 2)
        
        x_center_norm = (x_center - x_grid * x_span) / (x_span)
        y_center_norm = (y_center - y_grid * y_span) / (y_span)
        
        w = x2 - x1
        h = y2 - y1
        
        w_norm = w / imgsize[1]
        h_norm = h / imgsize[0]
        
        dims[y_grid, x_grid, :, :] = (x_center_norm, y_center_norm, w_norm, h_norm)
        
        grid[y_grid, x_grid] += 1
        
    tensor = np.concatenate((class_prob.ravel(), confidence.ravel(), dims.ravel()))
    
    return tensor
        


# In[12]:


def augment_data(label, frame, imgsize = (224, 224), folder = 'D://Dataset/apperance_expressions'):
    # 이미지 파일 이름과 프레임을 가져옴, 이미지의 HSV 공간에서 임의로 SV 값을
    #스케일링, 변환, 조정함
    #새 이미지의 경계상자와 일치하도록 'frame'의 좌표를 조정함
    img = cv2.imread(folder + label)
    img = cv2.resize(img, imgsize)
    rows, cols = img.shape[:2]
    
    #translate_factor
    tr = np.random.random() * 0.2 + 0.01
    tr_y = np.random.randint(rows * -tr, rows * tr)
    tr_x = np.random.randint(cols * -tr, cols * tr)
    
    #scale_factor
    sc = np.random.random() * 0.4 + 0.8
    
    #이미지 포화 조정
    r = np.random.rand()
    
    if r < 0.5:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        fs = np,random.random() + 0.7
        fv = np.random.random() + 0.2
        img[:, :, 1] *= fs
        img[:, :, 2] *= fv
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        
    r = np.random.rand()
    
    if r < 0.3:
        M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        img = cv2.warpAffine(img, M, (cols, rows))
        frame = coord_translate(frame, tr_x, tr_y)
    elif r < 0.6:
        # 동일한 크기로 이미지 확장
        placeholder = np.zeros_like(img)
        meta = cv2.resize(img, (0, 0), fx = sc, fy = sc)
        
        if sc < 1:
            placeholder[:meta.shape[0], :meta.shape[1]] = meta
        else:
            placeholder = meta[:placeholder.shape[0], :placeholder.shape[1]]
        img = placeholder
        frame = coord_scale(frame, sc)
        
    return img, frame
            
        
    


# In[13]:


def generator(label_keys, label_frames, batch_size = 64, folder = 'D://Dataset/apperance_expressions'):
    # test data와 train data split
    num_samples = len(label_keys)
    indx = label_keys
    
    while 1:
        shuffle(indx)
        for offset in range(0, num_samples, batch_size):
            batch_samples = indx[offset:offset + batch_size]
            
            images = []
            gt = []
            
            for batch_sample in batch_samples:
                im, frame = augument_data(batch_sample, label_frames[batch_sample])
                im = im.astype(np.float32)
                im -= 128
                images.append(im)
                frame_tensor = label_to_tensor(frame)
                gt.append(frame_tensor)
                
            X_train = np.array(images)
            y_train = np.array(gt)
            yield shuffle(X_train, y_train)


# In[14]:


def plot_history(history_object): # epoch에 따른 mean squared error 그래프
    print(history_object.history.keys())
    
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc = 'upper right')
    plt.show()


# In[15]:


if __name__ == '__main__':
    weights_path = 'imagenet'
    save_prefix = 'run1_'
    learning_rate = 1e-2

    if len(sys.argv) > 3:
        weights_path = sys.argv[1]
        save_prefix = sys.argv[2]
        learning_rate = float(sys.argv[3])
    elif len(sys.argv) > 2:
        weights_path = sys.argv[1]
        save_prefix = sys.argv[2]
    elif len(sys.argv) > 1:
        weights_path = sys.argv[1]
    
    model = ResNet50(include_top = False, load_weight = True, 
                     weights = weights_path, input_shape = (224, 224, 3))
    
    with open('label_frames.p', 'rb') as f:
        label_frames = pickle.load(f)

    label_keys = list(label_frames.keys())
    lbl_train, lbl_validn = train_test_split(label_keys, test_size = 0.2)

    train_generator = generator(lbl_train, label_frames)
    validation_generator = generator(lbl_validn, label_frames)

    optimizer = Adam(lr = 0.001)
    model.compile(optimizer = optimizer, loss = custom_loss)
    history = model.fit_generator(train_generator, validation_data = validation_generator, 
                   steps_per_epoch = len(lbl_train) // 64, epochs = 15,
                   validation_steps = len(lbl_validn) // 64
                   )

    

