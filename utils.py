import numpy as np
import torch
import pickle
import sys,os
import pandas as pd
import matplotlib.pyplot as plt

# This divides session 1,2,3 and returns train,test list

def load_cv_data(ch_dir, filename, csv_name, cv_number):
    csv_list = pd.read_csv(csv_name)

    file_list = os.listdir(ch_dir)
    file_list.sort()
    
    tester = filename[0:2]
    if '_' in tester:
        tester = tester[0]
    
    list_1 = [0,1,2,3,4,15,16,17,18,19,30,31,32,33,34]
    list_2 = [5,6,7,8,9,20,21,22,23,24,35,36,37,38,39]
    list_3 = [10,11,12,13,14,25,26,27,28,29,40,41,42,43,44]
    
    if cv_number == 1:
        print('#1 as test, preparing data')
        train_list = list_2 + list_3
        test_list = list_1
    elif cv_number == 2:
        print('#2 as test, preparing data')
        train_list = list_1 + list_3
        test_list = list_2
    else:
        print('#3 as test, preparing data')
        train_list = list_1 + list_2
        test_list = list_3

    train_result = []
    train_label = []
    test_result = []
    test_label = []
    
    for i,row in csv_list.iterrows():
        file_part = row['data'].split('_')
        name = row['data']
        label = float(row['label'])
        movie_num = int(file_part[2].split('-')[0])
        if file_part[0] == tester:
            if movie_num in train_list:
                train_result.append(name)
                train_label.append(label)
            else:
                test_result.append(name)
                test_label.append(label)
        else:
            continue

    train_all = np.vstack((train_result, train_label))
    test_all = np.vstack((test_result, test_label))
    
    return train_all, test_all

# Normalise DE features from all participants

def normalise(path, new_path):

    file_list = os.listdir(path)
    flattened_data = []
    first = True

    if not os.path.exists(new_path):
         os.mkdir(new_path)

    for i in file_list:
         data_npz = np.load(path + i)
         data = pickle.loads(data_npz['data'])
         for j in range(len(data)):
                 if j == 0 and first:
                      flattened_data = data[j]
                      first = False
                 else:
                    flattened_data = np.vstack((flattened_data,data[j]))

    flattened_data = np.array(flattened_data)

    mean = np.mean(flattened_data)
    std = np.std(flattened_data)
    print("mean:{:.3f}, std:{:.3f}".format(mean,std))

    for i in file_list:
          data_dict = {}
          label_dict = {}
          data_npz = np.load(path + i)
          data = pickle.loads(data_npz['data'])
          label = pickle.loads(data_npz['label'])

          for j in range(len(data)):
            eeg_data = np.array(list(data[j]))
            norm_eeg_data = (eeg_data - mean) / std
            data_dict[j] = norm_eeg_data
            label_dict[j] = label[j]
          data_pickle = pickle.dumps(data_dict)
          label_pickle = pickle.dumps(label_dict)

          np.savez(new_path+i,data=data_pickle,label=label_pickle)

# making csv file for data loader
def make_csv(eeg_dir, name):
    file_list = os.listdir(eeg_dir)
    file_list.sort()
    data_name = []
    label_name = []
    # f_id -> participant's number
    for f_id in file_list:
        if f_id.endswith('.npy'):
        # extract data and label from file name
            data = f_id.split('.')[0]+'.'+f_id.split('.')[1]
            label = float(data.split('_')[1])
        
        # add data & label into list
            data_name.append(data)
            label_name.append(label)
    df = pd.DataFrame({'data': data_name, 'label': label_name})
    df.to_csv(os.path.join('./',name), index=False)

"""
confusion matrix functions

"""
def create_confusion_matrix(net, device, testloader, num_classes):
    net.eval()
    
    conf_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data
            inputs = inputs.to(device=device,dtype=torch.float)
            targets = np.asarray(targets)
            targets = targets.astype(float)
            targets = torch.from_numpy(targets).to(device=device)
            targets = targets.type(torch.LongTensor).to(device=device)
            
            outputs = net(inputs)
            _,predicted = torch.max(outputs.data, 1)
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
    
    conf_matrix_norm = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)
    conf_matrix_percent = conf_matrix_norm * 100
    
    return conf_matrix.numpy(), conf_matrix_percent.numpy()

def plot_confusion_matrix(conf_matrix, num_classes, title, cmap=plt.cm.Blues):
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    
    thresh = conf_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, int(conf_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def create_and_save_confusion_matrix(net, testloader, num_classes, title, path, device, cmap=plt.cm.Blues):
    conf_matrix, conf_matrix_percent = create_confusion_matrix(net, device, testloader, num_classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(conf_matrix, num_classes, title, cmap=cmap)
    fig.savefig(path + '_confusion_matrix.png')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(conf_matrix_percent, num_classes, title + ' (Percent)', cmap=cmap)
    fig.savefig(path + '_confusion_matrix_percent.png')


# Normalise all DE features
path = './SEED-V/EEG_DE_features/'
new_path = './SEED-V/Normalized_EEG/'

normalise(path, new_path)
