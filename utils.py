import numpy as np
from PIL import Image
import math
#import mat73
import scipy.io as sio
import time
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import os

def rgb2ycbcr(rgb_img):
    ycbcr_img = np.zeros(rgb_img.shape)
    
    mat = np.array(
        [[ 65.481, 128.553, 24.966 ],
         [-37.797, -74.203, 112.0  ],
         [  112.0, -93.786, -18.214]])
    offset = np.array([16, 128, 128])
    
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img

def imread(imgName,block_size):
    Iorg = np.array(Image.open(imgName), dtype='float32')
    if Iorg.ndim==3:
        img_rec_name = "%s_groundtruth.png" % (imgName)
        if not os.path.exists(img_rec_name):
            Iorg = rgb2ycbcr(Iorg)
            Iorg = np.array(Iorg[:,:,0], dtype='float32')
            Iorg_save = Image.fromarray(Iorg.astype(np.uint8))
            Iorg_save.save(img_rec_name)
        else:
            Iorg = np.array(Image.open(img_rec_name), dtype='float32')
    [row, col] = Iorg.shape
    if np.mod(col,block_size) > 0:
        col_pad = block_size-np.mod(col,block_size)
        Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    else:
        col_pad = 0
        Ipad = Iorg
    if np.mod(row,block_size) > 0:
        row_pad = block_size-np.mod(row,block_size)
        Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape
    return [Iorg, row, col, Ipad, row_new, col_new]

def img2col(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = int(np.ceil(row/block_size))
    col_block = int(np.ceil(col/block_size))
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row, block_size):
        for y in range(0, col, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            count = count + 1
    return img_col

def col2img(X_col, row, col, row_new, col_new,block_size):
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def load_train(file_name, quantity):
    print('Loading Training Data ...')
    time_load_data_begin = time.time()
    Training_data = []
    
    for quan in range(1,quantity+1,1):
        data_labels = sio.loadmat(file_name)
        data = data_labels['labels']
        if quan == 1:
            Training_data = data
        else:
            Training_data = np.concatenate((Training_data, data), axis=0)
    
    num_of_data = Training_data.shape[0]
    time_load_data = time.time() - time_load_data_begin
    print('Loading Training Data use {0:<.4f} sec'.format(time_load_data))

    return [num_of_data, Training_data]

def load_CS_output(data_dir, quantity, b_size, stride):
    print('Loading CS output Data ...')
    time_load_data_begin = time.time()
    CS_output_data = []
    
    for quan in range(1,quantity+1,1):
        file_name = [data_dir + 'train_data_' + str(b_size) + '_stride_' + str(stride) + '_output_of_CS_block_' + str(quan) + '.npy'][0]
        data = np.load(file_name)
        if quan == 1:
            CS_output_data = data
        else:
            CS_output_data = np.concatenate((CS_output_data, data), axis=0)
    
    time_load_data = time.time() - time_load_data_begin
    print('Loading CS output Data use {0:<.4f} sec'.format(time_load_data))

    return CS_output_data

def load_Set11():
    im_name = ['barbara','boats','cameraman','fingerprint',
           'flinstones','foreman','house','lena256',
           'Monarch','Parrots','peppers256']
    filepaths = []
    for i in range(11):
        filepaths += glob.glob('./Test_Image/' + im_name[i] + '.tif')
    ImgNum = len(filepaths)
    
    return [ImgNum, filepaths]

def img_1_save(gt,recon,psnr,ssim,time,filename,is_print_img=True):
    name = 'reconstruction, PSNR=%.2f, SSIM=%.4f, time=%.2f' % (psnr,ssim,time)
    plt.title(name)
    recon_plt = Image.fromarray(np.clip(recon, 0, 255).astype(np.uint8))
    recon_plt.save(filename)

def img_2_save(gt,recon,psnr,ssim,time,filename,is_print_img=True):
    plt.figure(figsize=(12,6))
    
    plt.subplot(121)
    plt.title('ground-truth')
    gt_plt = Image.fromarray(gt.astype(np.uint8))
    plt.imshow(gt_plt,cmap='gray')
    
    plt.subplot(122)
    name = 'reconstruction, PSNR=%.2f, SSIM=%.4f, time=%.2f' % (psnr,ssim,time)
    plt.title(name)
    recon_plt = Image.fromarray(np.clip(recon, 0, 255).astype(np.uint8))
    plt.imshow(recon_plt,cmap='gray')
    
    plt.savefig(filename)
    if is_print_img == True:
        plt.show()

def img_3_save(gt,CS_rec,CS_PSNR,CS_SSIM,DB_rec,DB_PSNR,DB_SSIM,filename):
    plt.figure(figsize=(18,6))
    plt.subplot(131)
    plt.title('ground-truth')
    gt_plt = Image.fromarray(gt.astype(np.uint8))
    plt.imshow(gt_plt,cmap='gray')
    
    plt.subplot(132)
    name = 'CS reconstruction, PSNR=%.3f, SSIM=%.4f' % (CS_PSNR,CS_SSIM)
    plt.title(name)
    cor_plt = Image.fromarray(np.clip(CS_rec, 0, 255).astype(np.uint8))
    plt.imshow(cor_plt,cmap='gray')
    
    plt.subplot(133)
    name = 'after deblocking, PSNR=%.3f, SSIM=%.4f' % (DB_PSNR,DB_SSIM)
    plt.title(name)
    recon_plt = Image.fromarray(np.clip(DB_rec, 0, 255).astype(np.uint8))
    plt.imshow(recon_plt,cmap='gray')
    
    plt.savefig(filename)
    plt.show()

def pad_to_large(img,block_size):
    [row, col] = img.shape
    if col < block_size:
        col_pad = block_size-col
        img_pad = np.concatenate((img, np.zeros([row, col_pad])), axis=1)
    else:
        col_pad = 0
        img_pad = img
    if row < block_size:
        row_pad = block_size-row
        img_pad = np.concatenate((img_pad, np.zeros([row_pad, col+col_pad])), axis=0)
    return img_pad
            
def to_print(s,file):
    output_file = open(file, 'a')
    output_file.write(s)
    output_file.close()
    print(s,end='')
    
def reshape_33block_to_9row(X_input,block_size,block_num):
    bs = block_size
    bnsqrt = np.int32(np.sqrt(block_num))
    bsn = block_size*bnsqrt
    bsq = block_size*block_size
    
    X = tf.reshape(X_input,shape=[-1,bsn,bsn])
    row_vec_out = []

    for row in range(bnsqrt):
        for col in range(bnsqrt):
            Img_block = X[:,row*bs:(row+1)*bs, col*bs:(col+1)*bs]
            row_vec_out.append(tf.reshape(Img_block,shape=[-1,bsq]))
    return row_vec_out

def reshape_9row_to_33block(X_input,block_size):
    bs = block_size
    X = X_input
    
    b11 = tf.reshape(X[0],shape=[-1,bs,bs])
    b12 = tf.reshape(X[1],shape=[-1,bs,bs])
    b13 = tf.reshape(X[2],shape=[-1,bs,bs])
    b21 = tf.reshape(X[3],shape=[-1,bs,bs])
    b22 = tf.reshape(X[4],shape=[-1,bs,bs])
    b23 = tf.reshape(X[5],shape=[-1,bs,bs])
    b31 = tf.reshape(X[6],shape=[-1,bs,bs])
    b32 = tf.reshape(X[7],shape=[-1,bs,bs])
    b33 = tf.reshape(X[8],shape=[-1,bs,bs])
    
    row1 = tf.concat((b11,b12,b13),axis=2)
    row2 = tf.concat((b21,b22,b23),axis=2)
    row3 = tf.concat((b31,b32,b33),axis=2)
    
    block_out = tf.concat((row1,row2,row3),axis=1)

    return block_out

def reshape_4row_to_22block(X_input,block_size):
    bs = block_size
    X = X_input
    
    b11 = tf.reshape(X[0],shape=[-1,bs,bs])
    b12 = tf.reshape(X[1],shape=[-1,bs,bs])
    b21 = tf.reshape(X[2],shape=[-1,bs,bs])
    b22 = tf.reshape(X[3],shape=[-1,bs,bs])
    
    row1 = tf.concat((b11,b12),axis=2)
    row2 = tf.concat((b21,b22),axis=2)
    
    block_out = tf.concat((row1,row2),axis=1)

    return block_out



