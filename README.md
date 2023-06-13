# EEG-based Emotion Recognition using Multi-Channel Images with ResNet50

This project's objective is Emotion recognition of EEG data by converting 1D EEG features into 2D images and combining these images into Multi-channel trained by ResNet50.

## Dependencies
- python 3.10.7
- scipy 1.10.1
- pytorch 1.13.1
- numpy 1.24.1
- pandas 1.5.3
- matplotlib 3.7.1

## Dataset(SEED-V)
- [SEED-V: A dataset collection for various purposes using EEG signals](https://bcmi.sjtu.edu.cn/~seed/seed-v.html#)
## Usage
The sequence to run these code is "utils.py", "DataGenerator.py", and "ResNet50.py". The codes illustrated below are already implemented. Below explanation is in case, the data path is different with our experiment. 

### Naming
For example, one single waveband image is saved as "10_0.0_10-0a.npy" which means 'particiapnt_number'-'emotion_label'-'stimuli_number'-'time(second)''brain wave'.

Five channel (five waveband) images are saved similar way but except brainwave such as "10_0.0_10-0.npy".
13ch images are saved as "10_0.0_10-a.npy"
65ch images are saved as "10_0.0_10.npy"
### utils.py
"utils.py" contains the codes to create confusion matrix, normalise DE feature, make csv file for DataLoader, and split train and test sets

In utils.py, to normalise the data, change path of DE features and set new path to save normalised data. 
```python
path = './SEED-V/EEG_DE_features/'
new_path = './SEED-V/Normalized_EEG'

normalise(path, new_path)
```

### DataGenerator.py
"DataGenerator.py" contains the codes to generate EEG images from DE features to train ResNet50.

First, set Normalised EEG path at the start of the code.
```python
 eeg_dir = './SEED-V/Normalized_EEG/'
```

Then change the path to save generated images if necessary.
```python
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
```

### ResNet50.py
"ResNet50.py" contains the codes of our model. 

Change the paths, cv_num, and csv file names if necessary. cv_num is the test session number. If it is 3, then session 1,2 will be used to train,validate the model and session 3 will be test set.
```python
# data paths
norm5ch50dir = './data/5ch/50x50/normalize'
norm50dir = './data/1ch/50x50/normalize'
normtime50dir = './data/time_series/50x50/normalize(1wave)'
normtime5w50dir = './data/time_series/50x50/normalize(5wave)'

# csv files, cv_num = which session tobe test session
csv_name1 = 'eeg_image(1ch)_dataset.csv'
csv_name2 = 'eeg_image(5ch)_dataset.csv'
csv_name3 = 'eeg_image(time)_dataset.csv'
csv_name4 = 'eeg_image(5time)_dataset.csv'
cv_num = 3

"""
split and load train, test dataset

change data_path, csv_file to train other inputs, change cv_num to choose other session as test set

ex) train_all_tmp, test_all = load_eeg_data(participants,data_path,
                                            csv_file, cv_num)
"""
eeg_dir = './SEED-V/EEG_DE_features'
participants = os.listdir(eeg_dir)
train_all_tmp, test_all = load_eeg_data(participants,norm5ch50dir,csv_name2, cv_num)
```
Change the summary writer path if necessary and change 'chan' which dependent to input.
```python
writer = mkwriter('norm1_50')

net = ResNet50(block=Bottleneck, layers=[3,4,6,3],chan='5ch') # chan ='1ch','5ch','time','5time' 
net = net.to(device)
```
Change './train_history/norm1_50' to summary writer path.
```python
model_path = os.path.join('./train_history/norm1_50','best_model.pth')
```

Codes to create confusion matrix. Change model_path, net's 'chan' depend on your attempt.
```python
# create and save confusion matrix

model_path = './train_history/norm1_50/best_model.pth'
net = ResNet50(block=Bottleneck, layers=[3,4,6,3],chan='5ch')
net.load_state_dict(torch.load(model_path))
net = net.to(device)
utils.create_and_save_confusion_matrix(net=net,device=device, testloader=testloader, num_classes=5,
                                       title='Confusion Matrix',path=model_path, cmap=plt.cm.Blues)
```
