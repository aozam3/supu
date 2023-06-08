import numpy as np
import pandas
from srcs.utils import AlconDataset, AlconTargets,unicode_to_kana_list, evaluation
from srcs.myalgorithm import MyAlgorithm, VGG16, KuzushiData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import os
np.random.seed(42)
import matplotlib
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
import random
import torch.nn.functional as F
import torch
#import torchvision
from matplotlib import pyplot as plt
from torchvision import utils
import warnings
from PIL import Image
from torch.utils.data import Subset
from typing import Callable, Optional
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader

def show(img):
  # convert tensor to numpy array
  npimg = img.numpy()
  # Convert to H*W*C shape
  npimg_tr=np.transpose(npimg, (1,2,0))
  plt.imshow(npimg_tr,interpolation='nearest')
  plt.show()

# pathの準備
datapath='./dataset/'; 

# データの読み込み
targets = AlconTargets(datapath, train_ratio=0.99)
traindata = AlconDataset(datapath, targets.train, isTrainVal=True)
valdata = AlconDataset(datapath,targets.val, isTrainVal=True)
testdata = AlconDataset(datapath,targets.test, isTrainVal=False)

#print(traindata[1]) #(<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=57x327 at 0x7F81B2D470A0>, ['U+3066', 'U+3082', 'U+3068'])

print('Building model ...')
myalg = MyAlgorithm()   
myalg.build_model(traindata) 
x_train, y_train = myalg.build_model(traindata) 
print('done')

#print(x_train[0])
"""
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
plt.imshow(x_train[0], cmap='gray')
print(x_train[0])
plt.show()
"""
#exit()
# Datasetの長さ
num_train = len(x_train)
#num_test = len(x_test)

# x_train, x_test, y_train, y_testのデータ構造、最大値、最小値
#print(x_train)
#print(y_train)
#print(type(x_train), x_train.shape)
#print(type(y_train), y_train.shape)

# データの分割
train_idx = list(range(num_train))
random.shuffle(train_idx)

# trainデータの1割をパラメータ調整のためのvalidationに使用
val_frac = 0.1

# validationのデータ長の計算
num_val = int(num_train * val_frac) 
num_train = num_train - (num_val * 2)

test_idx = train_idx[num_train + num_val:]
val_idx = train_idx[num_train + 1 : num_train + num_val + 1]
train_idx = train_idx[:num_train]

x_test = x_train[num_train + num_val:]
y_test = y_train[num_train + num_val:]
x_valid = x_train[num_train + 1 : num_train + num_val + 1]
y_valid = y_train[num_train + 1 : num_train + num_val + 1]
x_train = x_train[:num_train]
y_train = y_train[:num_train]
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)

# データの変換（リサイズとデータ拡張のための水平方向のフリップ）
data_transform = transforms.Compose([
  transforms.Resize(32),
#  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
])

test_transform = transforms.Compose([
  transforms.Resize(32),
  transforms.ToTensor(),
])

# データの設定
train_set = KuzushiData(x_train, y_train, data_transform)
valid_set = KuzushiData(x_valid, y_valid, data_transform)
test_set = KuzushiData(x_test, y_test, test_transform)

# データの内容の確認のためのデータ変換処理
# numpy to tensor
img=torch.from_numpy(train_set.img.astype(np.uint8)).clone()
# add a dimension to tensor to become B*C*H*W
img=img.unsqueeze(1)

# 最初の40個のイメージを表示
x_grid=utils.make_grid(img[:40], nrow=8, padding=2)
print(x_grid.shape)
show(x_grid)

image, label = train_set[0]
print(image)
plt.imshow(image.squeeze(), cmap='gray')
plt.show()
print('Label:', label)

train_loader = DataLoader(train_set, batch_size=64, num_workers=2, shuffle=True, drop_last=True)
val_loader   = DataLoader(valid_set, batch_size=64, num_workers=2, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_set, batch_size=64, num_workers=2, shuffle=False, drop_last=False)

# Get an element from the dataset
test_x, _ = train_set[0] # each element of the dataset is a couple (image, label)
test_x = test_x.unsqueeze(dim=0)

# Create the model
model = VGG16((1,32,32), batch_norm=True)
output = model(test_x)
output.shape

torch.cuda.is_available()

dev = torch.device('cpu')
print(dev)

# Define an optimizier
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr = 0.01)
# Define a loss 
criterion = nn.CrossEntropyLoss()

def train(net, loaders, optimizer, criterion, epochs=20, dev=dev, save_param = False, model_name="valerio"):
    try:
        net = net.to(dev)
        #print(net)
        # Initialize history
        history_loss = {"train": [], "val": [], "test": []}
        history_accuracy = {"train": [], "val": [], "test": []}
        # Store the best val accuracy
        best_val_accuracy = 0

        # Process each epoch
        for epoch in range(epochs):
            # Initialize epoch variables
            sum_loss = {"train": 0, "val": 0, "test": 0}
            sum_accuracy = {"train": 0, "val": 0, "test": 0}
            # Process each split
            for split in ["train", "val", "test"]:
                if split == "train":
                  net.train()
                else:
                  net.eval()
                # Process each batch
                for (input, labels) in loaders[split]:
                    # Move to CUDA
                    input = input.to(dev)
                    labels = labels.to(dev)
                    # Reset gradients
                    optimizer.zero_grad()
                    # Compute output
                    pred = net(input)
                    #pred = pred.squeeze(dim=1) # Output shape is [Batch size, 1], but we want [Batch size]
                    #labels = labels.unsqueeze(1)
                    labels = labels.long()
                    loss = criterion(pred, labels)
                    # Update loss
                    sum_loss[split] += loss.item()
                    # Check parameter update
                    if split == "train":
                        # Compute gradients
                        loss.backward()
                        # Optimize
                        optimizer.step()
                    # Compute accuracy
                    #pred_labels = pred.argmax(1) + 1
                    #pred_labels = (pred >= 0.5).long() # Binarize predictions to 0 and 1
                    _,pred_label = torch.max(pred, dim = 1)
                    pred_labels = (pred_label == labels).float()

                    batch_accuracy = pred_labels.sum().item()/input.size(0)
                    # Update accuracy
                    sum_accuracy[split] += batch_accuracy
            # Compute epoch loss/accuracy
            epoch_loss = {split: sum_loss[split]/len(loaders[split]) for split in ["train", "val", "test"]}
            epoch_accuracy = {split: sum_accuracy[split]/len(loaders[split]) for split in ["train", "val", "test"]}

            # Store params at the best validation accuracy
            if save_param and epoch_accuracy["val"] > best_val_accuracy:
              #torch.save(net.state_dict(), f"{net.__class__.__name__}_best_val.pth")
              torch.save(net.state_dict(), f"{model_name}_best_val.pth")
              best_val_accuracy = epoch_accuracy["val"]

            # Update history
            for split in ["train", "val", "test"]:
                history_loss[split].append(epoch_loss[split])
                history_accuracy[split].append(epoch_accuracy[split])
            # Print info
            print(f"Epoch {epoch+1}:",
                  f"TrL={epoch_loss['train']:.4f},",
                  f"TrA={epoch_accuracy['train']:.4f},",
                  f"VL={epoch_loss['val']:.4f},",
                  f"VA={epoch_accuracy['val']:.4f},",
                  f"TeL={epoch_loss['test']:.4f},",
                  f"TeA={epoch_accuracy['test']:.4f},")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        # Plot loss
        plt.title("Loss")
        for split in ["train", "val", "test"]:
            plt.plot(history_loss[split], label=split)
        plt.legend()
        plt.show()
        # Plot accuracy
        plt.title("Accuracy")
        for split in ["train", "val", "test"]:
            plt.plot(history_accuracy[split], label=split)
        plt.legend()
        plt.show()

# Define dictionary of loaders
loaders = {"train": train_loader,
           "val": val_loader,
           "test": test_loader}

# Train model
train(model, loaders, optimizer, criterion, epochs=10, dev=dev)

# 推論のテスト
test_x, _ = test_set[0] # テストデータ
print(test_x)
test_x = test_x.unsqueeze(dim=0)
output = model(test_x.to(dev))
print(output)
print(output.shape)
_, predicted = torch.max(output.data, 1)
print(predicted) # 推論結果

# テストデータに対する推論
H = 10
W = 10
OFFSET = 0 # 最初から何番目のデータか
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.4, wspace=0.4)

for j in range(W):
  for i in range(H):
    img, label = test_set[OFFSET+j*H+i]
    img = img.unsqueeze(dim=0)
    output = model(img.to(dev))
    _, prediction = torch.max(output, 1)

#    print("pred:")
#    print(prediction)
#    print(prediction[0])
#    print(prediction[0].item())
#    print(type(prediction[0]))  
    
    plt.subplot(H, W, j*H+i+1)
    plt.imshow(x_test[OFFSET+j*H+i].reshape(28, 28), cmap='gray')

    if prediction.item() == label:
      plt.title(prediction[0].item(), fontsize=12, color = "green")
    else:
      plt.title("L:{} != P:{}".format(label, prediction.item()), fontsize=12, color = "red")
    
    plt.axis('off')

plt.show()

"""


# トレーニングモデル
print('Building model ...')
myalg = MyAlgorithm()  
myalg.build_model(traindata) 
#x_train, y_train = myalg.build_model(traindata) 
print('done')

# List of indexes on the training set
train_idx = list(range(num_train))

# Shuffle the training set
import random

random.shuffle(train_idx)
for i in range(10):
  print(train_idx[i])

"""

"""
#y_trainの整形
y_train2 = []
for i in range((len(y_train))):
    for j in range(3):
        y_train2.append(y_train[0][j])

#x_val, y_valの整形
N = len(valdata)
sheet = valdata.getSheet()  # Get initial sheet
x_val = []
y_val = []
for i in range(N):
    img,label = valdata[i]
    #print(valdata[i])
    img = np.array(img.convert('L'))
    feat = np.array(img.reshape(-1, img.size))
    a = np.zeros((3,200000))
    list = np.array_split(feat[0], 3)
    #print(list)
    #print(n[0], img.size, x_train[i][0]) 
    for j in range(len(list[0])):
        a[0][j] = list[0][j]

    for j in range(len(list[1])):
        a[1][j] = list[1][j]

    for j in range(len(list[2])):
        a[2][j] = list[2][j]

    for k in range(3):
        x_val.append(a[k])
        y_val.append(label[k])

#print(x_val)
#print(y_val)

"""


"""
#4-kNN
clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
clf.fit(x_train, y_train2)

# 評価
test_score = clf.score(np.array(x_val), y_val)
print('Test accuracy:', test_score)
"""
  
"""
#PCA + 4-kNN
pca = PCA(n_components= 60, random_state= 0 )
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_val)
clf = KNeighborsClassifier(n_neighbors= 4,weights='distance', n_jobs=-1)
clf.fit(x_train, y_train2)
p_test = clf.predict(x_test)
#print(np.array(p_test))

print(accuracy_score(y_val, p_test))

#added by description of K-49
mozi = []
for i in range(82):
    m = "U+30" + str(hex(66 + i)).replace('0x', '')
    mozi.append(m)

accs = []
for cls in mozi:
    mask = (np.array(y_val) == cls)
    #print([mask])
    #print((np.array(p_test) == cls))
    #print((np.array(p_test) == cls)[mask])
    n = (np.array(p_test) == cls)[mask]
    if len(n) == 0:
        cls_acc = 0.0
    else:
        cls_acc = n.mean()
    #print(cls_acc) 
    accs.append(cls_acc)
  
accs = np.mean(accs)
print('Test accuracy:', accs)
"""

"""
for i in range(N):
    # Prediction
    img,y_val = valdata[i]  # Get data
    y_pred = myalg.predict(img)  # Prediction
    #print('Prediction {}; GT {}; {}/{}'.format(
     #   unicode_to_kana_list(y_pred),unicode_to_kana_list(y_val),i,N))

    # Fill the sheet with y_pred
    sheet.iloc[i,1:4] = y_pred  


#評価
acc = evaluation( sheet, valdata.targets )
print('Accuracy = {:f} (%)'.format(acc))

# 予測した結果をCSVで保存.Zipで圧縮して提出
sheet.to_csv('test_prediction.csv',index=False)

"""