import scipy.interpolate as interp
import numpy as np
import pandas as pd
import pickle
import matplotlib.pylab as plab
import os
import pickle
import torchvision.transforms as transforms

import utils

eeg_dir = './SEED-V/Normalized_EEG/'


# divide 5 frequency waves from individual participant's one experiment
def five_wave(eeg_data):
    delta = [] 
    theta = []
    alpha = []
    beta = []
    gamma = []
    for i in range(len(eeg_data)):
        if i%5 == 0 :
            delta.append(eeg_data[:][i])
        elif i%5 == 1:
            theta.append(eeg_data[:][i])
        elif i%5 == 2:
            alpha.append(eeg_data[:][i])
        elif i%5 == 3:
            beta.append(eeg_data[:][i])
        else:
            gamma.append(eeg_data[:][i])
    return delta,theta,alpha,beta,gamma

# x,y coordinate to plot data on feature map
def dataarray(wave):
    example_matrix = np.zeros((9,9))
    tmp = wave.tolist()
    xi = [0,0,0,
          1,1,
          2,2,2,2,2,2,2,2,2,
          3,3,3,3,3,3,3,3,3,
          4,4,4,4,4,4,4,4,4,
          5,5,5,5,5,5,5,5,5,
          6,6,6,6,6,6,6,6,6,
          7,7,7,7,7,7,7,
          8,8,8,8,8
         ]
    yi = [3,4,5,
          2,6,
          0,1,2,3,4,5,6,7,8,
          0,1,2,3,4,5,6,7,8,
          0,1,2,3,4,5,6,7,8,
          0,1,2,3,4,5,6,7,8,
          0,1,2,3,4,5,6,7,8,
          1,2,3,4,5,6,7,
          2,3,4,5,6
         ]
    loc = list(zip(xi,yi))


    example_matrix = np.zeros((9,9))
    for i in range(len(tmp)):
        example_matrix[loc[i]] = tmp.pop(0)
            
        
    return example_matrix 

# plot data and interpolate scattered plots
def create50x50(wave):
    newz = np.ravel(dataarray(wave),order='C')
    
    xi = yi = [2,6,10,14,18,22,26,30,34]
    coordi = []
    for i in xi:
        for j in yi:
            coordi.append([i,j])
    X = Y = np.linspace(0,37)
    X,Y = np.meshgrid(X,Y)
    interpolator = interp.CloughTocher2DInterpolator(coordi,newz)
    Z = interpolator(X,Y)
    
    return Z.T
    
# save feature map to numpy
def eeg_to_image(wave, label, tester, wave_c, movie_num,res_dir):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    wave2 = np.array(wave).T

    for i in range(len(wave2)):       
        f_name = res_dir + '/' + str(tester) + '_' + str(label[i]) + '_' + str(movie_num) + '-' + str(i) + str(wave_c) #+ '.png'
        z = wave2.T
        step1 = create50x50(z[:,i])
        step1[np.isnan(step1)] = 0
        np.save(f_name,step1)

# generate EEG image of one individual
def gen50datas(eeg_dir, file_name, img_dir):
    data_npz = np.load(eeg_dir + file_name)
    tester = file_name[0:2]
    if '_' in tester:
        tester = tester[0]
    data = pickle.loads(data_npz['data'])
    label = pickle.loads(data_npz['label'])
    for i in range(len(data)):
        d,t,a,b,g = five_wave(data[i].T)
        eeg_to_image(d,label[i], tester, 'd', i+1,img_dir)
        eeg_to_image(t,label[i], tester, 't', i+1,img_dir)
        eeg_to_image(a,label[i], tester, 'a', i+1,img_dir)
        eeg_to_image(b,label[i], tester, 'b', i+1,img_dir)
        eeg_to_image(g,label[i], tester, 'g', i+1,img_dir)

    utils.make_csv(img_dir,'eeg_image(1ch)_dataset.csv')

# generate 5 frequency waves into one image with 5 channels
def to5channel_npy(file_name,res_dir,data_path,img_size):

    d = np.load(data_path + file_name + "d.npy")
    t = np.load(data_path + file_name + "t.npy")
    a = np.load(data_path + file_name + "a.npy")
    b = np.load(data_path + file_name + "b.npy")
    g = np.load(data_path + file_name + "g.npy")
    
    multi_channel_img = np.zeros((img_size,img_size,5)) # [chan, width, height]

    multi_channel_img[:,:,0] = d
    multi_channel_img[:,:,1] = t
    multi_channel_img[:,:,2] = a
    multi_channel_img[:,:,3] = b
    multi_channel_img[:,:,4] = g

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    final_file_name = res_dir + '/' + file_name + '.npy'
    np.save(final_file_name, multi_channel_img, allow_pickle=True)

    utils.make_csv(res_dir,'eeg_image(5ch)_dataset.csv')

# generate 1ch EEG image time-order concatenated image
def to1chTimewise_channel_npy(res_dir,data_path,img_size,csv_name):
    #list 0 to 12 time
    f_list = pd.read_csv(csv_name)
    file_name = f_list['data']
    data_result = []
    for i in file_name:
        num = i.split('-')[1]
        if int(num) > 12:
            continue
        else:
            data_result.append(i)
    data_result = np.array(data_result).reshape(720,13)
    new_data_result = np.hstack((data_result[:, :2], data_result[:, 5:], data_result[:, 2:5]))
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    multi_channel_img_d = np.zeros((img_size,img_size,13)) #first 13 sec
    multi_channel_img_t = np.zeros((img_size,img_size,13))
    multi_channel_img_a = np.zeros((img_size,img_size,13))
    multi_channel_img_b = np.zeros((img_size,img_size,13))
    multi_channel_img_g = np.zeros((img_size,img_size,13))

    for i in new_data_result:
        tmp_name = i[0].split('-')[0]       
            
        for j in range(len(i)):
            multi_channel_img_d[:,:,j] = np.load(data_path + i[j] + 'd.npy')
            multi_channel_img_t[:,:,j] = np.load(data_path + i[j] + 't.npy')
            multi_channel_img_a[:,:,j] = np.load(data_path + i[j] + 'a.npy')
            multi_channel_img_b[:,:,j] = np.load(data_path + i[j] + 'b.npy')
            multi_channel_img_g[:,:,j] = np.load(data_path + i[j] + 'g.npy')
            
        d_name = res_dir + '/' + tmp_name + '-d.npy'
        t_name = res_dir + '/' + tmp_name + '-t.npy'
        a_name = res_dir + '/' + tmp_name + '-a.npy'
        b_name = res_dir + '/' + tmp_name + '-b.npy'
        g_name = res_dir + '/' + tmp_name + '-g.npy'
            
        np.save(d_name, multi_channel_img_d, allow_pickle=True)
        np.save(t_name, multi_channel_img_t, allow_pickle=True)
        np.save(a_name, multi_channel_img_a, allow_pickle=True)
        np.save(b_name, multi_channel_img_b, allow_pickle=True)
        np.save(g_name, multi_channel_img_g, allow_pickle=True)

    utils.make_csv(res_dir,'eeg_image(time)_dataset.csv')

# generate 5ch EEG image time-order concatenated image
def to5chTimewise_channel_npy(res_dir,data_path,img_size,csv_name):
    #list 0 to 12 time
    f_list = pd.read_csv(csv_name)
    file_name = f_list['data']
    data_result = []
    for i in file_name:
        num = i.split('-')[1]
        if int(num) > 12:
            continue
        else:
            data_result.append(i)
    data_result = np.array(data_result).reshape(720,13)
    new_data_result = np.hstack((data_result[:, :2], data_result[:, 5:], data_result[:, 2:5]))
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    multi_channel_img = np.zeros((img_size,img_size,65)) #first 13 sec x 5

    for i in new_data_result:
        tmp_name = i[0].split('-')[0]
            
        for j in range(len(i)):
            multi_channel_img[:,:,j*5:(j+1)*5] = np.load(data_path + i[j] + '.npy')
            
        name = res_dir + '/' + tmp_name + '.npy'
            
        np.save(name, multi_channel_img, allow_pickle=True)
    
    utils.make_csv(res_dir,'eeg_image(5time)_dataset.csv')


# create single wave band data
file_list = os.listdir(eeg_dir)
for i in file_list:
    gen50datas(eeg_dir,i,'./data/1ch/50x50/normalize/')

# create five wave band data
f_list = pd.read_csv('eeg_image(1ch)_dataset.csv')
file_name = list(set([d[:-1] for d in f_list['data']]))
file_name.sort()
for i in file_name:
    to5channel_npy(i,'./data/5ch/50x50/normalize','./data/1ch/50x50/normalize/',50)

# create single wave band time order concatenated data
to1chTimewise_channel_npy('./data/time_series/50x50/normalize(1wave)',
                           './data/1ch/50x50/normalize/',50,
                           'eeg_image(5ch)_dataset.csv')

# create five wave band time order concatenated data
to5chTimewise_channel_npy('./data/time_series/50x50/normalize(5wave)',
                          './data/5ch/50x50/normalize/',50,'eeg_image(5ch)_dataset.csv')