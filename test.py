import tensorflow as tf
import numpy as np
import os
import time
import glob
import utils

### user define begin
Training_data_file_name = './training_data/BSDS500_train.mat'
CS_ratio = 25 # 25 means measurement rate 25%
### user define end


num_data_block = 8 # setting of training data
is_test = True

begin_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
q = 0.5
layer_num = 9

block_size = 32
n_output = 4096
n_output_of_A = 1024
n_input = np.int32(np.round(n_output_of_A*CS_ratio/100))

if is_test == True:
    ckpt_model_number = 100

block_size_large = 64
block_num = 4
batch_size = 64
learning_rate = 0.0001
EpochNum = 300
    
X_output = tf.compat.v1.placeholder(tf.float32, [None, n_output])

def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.compat.v1.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random.normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]

def add_fc(shape1, order_no):
    AA = tf.compat.v1.get_variable(shape=shape1, initializer=tf.contrib.layers.xavier_initializer(), name='FC_%d' % order_no)
    return AA

def ista_block(input_layers, QY,ATA):
    step_size = tf.Variable(1e-1, dtype=tf.float32)
    alpha = tf.Variable(1e-5, dtype=tf.float32)
    beta = tf.Variable(1.0, dtype=tf.float32)
    
    X = tf.reshape(input_layers[-1],shape=[-1,block_size_large,block_size_large])
    X_row = utils.reshape_33block_to_9row(X,block_size,block_num) # size (?,1024) list9

    GD = []
    for ii in range(block_num):
        temp = tf.add(X_row[ii] - tf.scalar_mul(step_size, tf.matmul(X_row[ii], ATA)), tf.scalar_mul(step_size, QY[ii]))    
        GD.append(temp)
    
    x1 = utils.reshape_4row_to_22block(GD,block_size)
    x2 = tf.reshape(x1, shape=[-1, block_size_large, block_size_large, 1])

    [Weights0, bias0] = add_con2d_weight_bias([3, 3, 1, 32], [32], 0)
    [Weights1, bias1] = add_con2d_weight_bias([3, 3, 32, 32], [32], 1)
    [Weights2, bias2] = add_con2d_weight_bias([3, 3, 32, 32], [32], 2)
    [Weights3, bias3] = add_con2d_weight_bias([3, 3, 32, 32], [32], 3)
    [Weights4, bias4] = add_con2d_weight_bias([3, 3, 32, 32], [32], 4)
    [Weights5, bias5] = add_con2d_weight_bias([3, 3, 32, 32], [32], 5)
    [Weights6, bias6] = add_con2d_weight_bias([3, 3, 32, 32], [32], 6)
    [Weights7, bias7] = add_con2d_weight_bias([3, 3, 32, 1], [1], 7)
    
    x30 = tf.nn.conv2d(x2, Weights0, strides=[1, 1, 1, 1], padding='SAME')
    x31 = tf.nn.relu(x30)
    x32 = tf.nn.conv2d(x31, Weights1, strides=[1, 1, 1, 1], padding='SAME')
    x33 = tf.nn.relu(x32)
    x34 = tf.nn.conv2d(x33, Weights2, strides=[1, 1, 1, 1], padding='SAME')
    x35 = tf.nn.relu(x34)
    x40 = tf.nn.conv2d(x35, Weights3, strides=[1, 1, 1, 1], padding='SAME')
    
    trun_param = alpha / ((0.1 + tf.abs(x40))**(1-q))
    x50 = tf.multiply(tf.sign(x40), tf.nn.relu(tf.abs(x40) - trun_param))
    x51 = x50 - x40
    
    x60 = tf.nn.conv2d(x51, Weights4, strides=[1, 1, 1, 1], padding='SAME')
    x61 = tf.nn.relu(x60)
    x62 = tf.nn.conv2d(x61, Weights5, strides=[1, 1, 1, 1], padding='SAME')
    x63 = tf.nn.relu(x62)
    x64 = tf.nn.conv2d(x63, Weights6, strides=[1, 1, 1, 1], padding='SAME')
    x65 = tf.nn.relu(x64)
    x66 = tf.nn.conv2d(x65, Weights7, strides=[1, 1, 1, 1], padding='SAME')

    x70 = x2 + beta * x66
    x80 = tf.reshape(x70, shape=[-1, n_output])
    
    x41 = tf.nn.conv2d(x40, Weights4, strides=[1, 1, 1, 1], padding='SAME')
    x42 = tf.nn.relu(x41)
    x43 = tf.nn.conv2d(x42, Weights5, strides=[1, 1, 1, 1], padding='SAME')
    x44 = tf.nn.relu(x43)
    x45 = tf.nn.conv2d(x44, Weights6, strides=[1, 1, 1, 1], padding='SAME')
    x46 = tf.nn.relu(x45)
    x47 = tf.nn.conv2d(x46, Weights7, strides=[1, 1, 1, 1], padding='SAME')
    x48 = x47 - x2
    
    return [x80, x48]



def inference_ista_deblock(layer_num, X_output, reuse):
    X1 = X_output
    AT = add_fc([n_output_of_A,n_input], 0)
    ATT = add_fc([n_input,n_output_of_A], 1)
    
    X2 = tf.reshape(X1,shape=[-1,block_size_large,block_size_large])
    X_9row = utils.reshape_33block_to_9row(X2,block_size,block_num)
    
    QY = []
    for ii in range(block_num):
        Y_temp = tf.matmul(X_9row[ii], AT)
        QY_temp = tf.matmul(Y_temp, ATT)
        QY.append(QY_temp)
    
    ATA = tf.matmul(AT,ATT)
    
    layers = []
    layers_symetric = []
    layers.append(QY)
    for i in range(layer_num):
        with tf.compat.v1.variable_scope('conv_%d' %i, reuse=reuse):
            [conv1, conv1_sym] = ista_block(layers, QY,ATA)
            layers.append(conv1)
            layers_symetric.append(conv1_sym)
    return [layers, layers_symetric]

[Prediction, Pre_symetric_CS] = inference_ista_deblock(layer_num, X_output, reuse=False)

def compute_cost(Prediction, X_output, layer_num):
    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))
    cost_sym = 0
    for k in range(layer_num):
        cost_sym += tf.reduce_mean(tf.square(Pre_symetric_CS[k]))

    return [cost, cost_sym]

[cost, cost_sym] = compute_cost(Prediction, X_output, layer_num)
cost_all = cost + 0.01 * cost_sym

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
optm = optimizer.minimize(cost_all)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

print("...............................")
print("Phase Number is %d, CS ratio is %d%%" % (layer_num, CS_ratio))
print("...............................\n")

model_dir = 'Trained_Model_layer_%d_ratio_%d' % (layer_num, CS_ratio)

saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=300)
sess = tf.compat.v1.Session(config=config)

if is_test == False:
    sess.run(tf.compat.v1.global_variables_initializer())     ##################### comment in testing
    output_file_name_log = "Log_output_%s.txt" % (model_dir)
    output_file_name_PSNR = "PSNR_Results_%s.txt" % (model_dir)
    output_file_name_loss = "Loss_Records_%s.txt" % (model_dir)

    [ImgNum, Training_data] = utils.load_train(Training_data_file_name, num_data_block)

    recon_dir = './Reconstruction_results/'
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)
    
    for epoch_i in range(1, EpochNum+1, 1):
        epoch_i_time_begin = time.time()
        rand_idx_all = np.random.permutation(ImgNum)
        for batch_i in range(ImgNum // batch_size):
            print('\rtraining epoch {0}, batch {1}/{2}'.format(epoch_i,batch_i+1, ImgNum//batch_size),end='')
            rand_idx = rand_idx_all[batch_i*batch_size:(batch_i+1)*batch_size]
            batch_data = Training_data[rand_idx, :].astype(np.float32)
            
            batch_begin = time.time()
            rec_row = sess.run(Prediction[-1], feed_dict={X_output: batch_data})
            batch_using_time = time.time() - batch_begin
            
            cost_value = sess.run(cost_all, feed_dict={X_output: batch_data})
            sess.run(optm, feed_dict={X_output: batch_data})
            
        epoch_i_using_time = time.time() - epoch_i_time_begin
        print('')
        print('epoch {0} spend {1:<.3f} sec'.format(epoch_i,epoch_i_using_time))
        
        Test_Img = './Test_Image_Set11'
        filepaths = glob.glob(Test_Img + '/*.tif')
        
        PSNR_all = np.zeros(len(filepaths))
        SSIM_all = np.zeros(len(filepaths))
        
        for img_i in range(len(filepaths)):
            [Iorg, row, col, Ipad, row_new, col_new] = utils.imread(filepaths[img_i],block_size_large)
            Irow = utils.img2col(Ipad, block_size_large).transpose()/255
            
            recon_start = time.time()
            rec_row = sess.run(Prediction[-1], feed_dict={X_output: Irow})
            recon_using_time = time.time() - recon_start
            
            X_rec = utils.col2img(rec_row.transpose(), row, col, row_new, col_new, block_size_large)
            rec_255 = np.clip(X_rec * 255, 0, 255)
            rec_PSNR = utils.psnr(rec_255, Iorg)
            temp = Iorg.shape
            rec_SSIM = sess.run(tf.image.ssim(tf.image.convert_image_dtype(tf.reshape(rec_255, shape=list(temp) + [1]), tf.float32), tf.reshape(Iorg, shape=list(temp) + [1]), max_val=255.0))
            
            out0 = 'img no %d: %s\n' %(img_i,filepaths[img_i].split('\\')[-1].split('.')[0])
            utils.to_print(out0,output_file_name_log)
            out0 = '    PSNR = %.3f, SSIM = %.4f, time %.2f\n' %(rec_PSNR,rec_SSIM,recon_using_time)
            utils.to_print(out0,output_file_name_log)
            
            PSNR_all[img_i] = rec_PSNR
            SSIM_all[img_i] = rec_SSIM
            
            img_name = filepaths[img_i].split('\\')[-1].split('.')[0]
            img_rec_name1 = [recon_dir + 'layer_' + str(layer_num) + '_ratio_' + str(CS_ratio)][0]
            img_rec_name2 = '_%s_epoch_%d_PSNR_%.3f_SSIM_%.4f_time_%.2f.png' % (img_name, epoch_i, rec_PSNR, rec_SSIM, recon_using_time)
            img_rec_name = [img_rec_name1 + img_rec_name2][0]
            utils.img_2_save(Iorg,rec_255,rec_PSNR,rec_SSIM,recon_using_time,img_rec_name)
        
        PSNR_avg = np.mean(PSNR_all)
        SSIM_avg = np.mean(SSIM_all)
        
        out0 = '\n epoch no %d\n' %(epoch_i)
        utils.to_print(out0,output_file_name_log)
        utils.to_print(out0,output_file_name_PSNR)
        out0 = '   avg PSNR {0:.3f}, SSIM {1:.4f}\n'.format(PSNR_avg,SSIM_avg)
        utils.to_print(out0,output_file_name_log)
        utils.to_print(out0,output_file_name_PSNR)
            
        saver.save(sess, './%s/Saved_Model_epoch_%d.ckpt' % (model_dir, epoch_i), write_meta_graph=False)
        out0 = "Run time of epoch %d is %.2f, PSNR is %.3f, SSIM is %.4f, loss is %.4f\n" % (epoch_i, epoch_i_using_time, PSNR_avg, SSIM_avg, cost_value)
        utils.to_print(out0,output_file_name_log)
        print()
            
        out0 = "[%02d/%02d] cost: %.5f, using time: %.3f sec\n" % (epoch_i, EpochNum, cost_value, epoch_i_using_time)
        utils.to_print(out0,output_file_name_loss)
        print("Training Finished")

else:
    saver.restore(sess, './%s/Saved_Model_epoch_%d.ckpt' % (model_dir, ckpt_model_number))
    output_file_name_test_ind = "./test_results/PSNR_testing_Results_ind_%s.txt" % (model_dir)
    output_file_name_test = "./test_results/PSNR_testing_Results_%s.txt" % (model_dir)
    
    is_print_img = True
    
    for DataSet in range(5):
        if DataSet == 0:
            Test_dataset,file_type = 'Set11', 'tif'
        elif DataSet == 1:
            Test_dataset,file_type = 'Set5', 'bmp'
        elif DataSet == 2:
            Test_dataset,file_type = 'Set14', 'bmp'
        elif DataSet == 3:
            Test_dataset,file_type = 'BSD68', 'png'
        elif DataSet == 4:
            Test_dataset,file_type = 'BSD100', 'jpg'
    
        if is_print_img == True:
            Test_Img_individual = ['../Test_Image_' + Test_dataset][0]
            filepaths = glob.glob(Test_Img_individual + '/*.' + file_type)
        
        test_dir = ['./test_results/' + Test_dataset + '_full' + '/'][0]
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            
        out0 = "%s:\n" % (Test_dataset)
        utils.to_print(out0,output_file_name_test_ind)
        utils.to_print(out0,output_file_name_test)
        
        PSNR_all = np.zeros(len(filepaths))
        SSIM_all = np.zeros(len(filepaths))
        
        for img_i in range(len(filepaths)):
            [Iorg, row, col, Ipad, row_new, col_new] = utils.imread(filepaths[img_i],block_size_large)
            Irow = utils.img2col(Ipad, block_size_large).transpose()/255
            
            recon_start = time.time()
            rec_row = sess.run(Prediction[-1], feed_dict={X_output: Irow})
            recon_using_time = time.time() - recon_start
            
            X_rec = utils.col2img(rec_row.transpose(), row, col, row_new, col_new, block_size_large)
            rec_255 = np.clip(X_rec * 255, 0, 255)
            rec_PSNR = utils.psnr(rec_255, Iorg)
            temp = Iorg.shape
            rec_SSIM = sess.run(tf.image.ssim(tf.image.convert_image_dtype(tf.reshape(rec_255, shape=list(temp) + [1]), tf.float32), tf.reshape(Iorg, shape=list(temp) + [1]), max_val=255.0))
            
            out0 = '  img no %d: %s\n' %(img_i,filepaths[img_i].split('\\')[-1].split('.')[0])
            utils.to_print(out0,output_file_name_test_ind)
            out0 = '      PSNR = %.3f, SSIM = %.4f, time %.4f\n' %(rec_PSNR,rec_SSIM,recon_using_time)
            utils.to_print(out0,output_file_name_test_ind)
            
            PSNR_all[img_i] = rec_PSNR
            SSIM_all[img_i] = rec_SSIM
            
            if is_print_img == True:
                img_name = filepaths[img_i].split('\\')[-1].split('.')[0]
                img_rec_name1 = '%s_PSNR_%.3f_SSIM_%.4f_time_%.4f.png' % (img_name, rec_PSNR, rec_SSIM, recon_using_time)
                img_rec_name = [test_dir + img_rec_name1][0]
                utils.img_1_save(Iorg,rec_255,rec_PSNR,rec_SSIM,recon_using_time,img_rec_name,is_print_img=False)
        
        PSNR_avg = np.mean(PSNR_all)
        SSIM_avg = np.mean(SSIM_all)
        
        out0 = "avg PSNR is %.3f, SSIM is %.4f\n\n" % (PSNR_avg, SSIM_avg)
        utils.to_print(out0,output_file_name_test_ind)
        utils.to_print(out0,output_file_name_test)
    print("Test Finished")

sess.close()
