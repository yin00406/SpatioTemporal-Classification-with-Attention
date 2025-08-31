import sys
import config
import os
import numpy as np
import random
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import pickle
import model
import utils
import pandas as pd
import math

# PARAMETERS
Switch_splitDataset = False
batch_size = 256
learning_rate = 1e-5
partition_size = batch_size*10
patchSize = 64

def partitions(image_patches, label_patches, partition_size, rand=True):
    if rand:
        index = random.sample(range(image_patches.shape[0]), image_patches.shape[0])
        image_patches = image_patches[index]
        label_patches = label_patches[index]
    num_partitions = math.ceil(image_patches.shape[0]/partition_size)
    image_patches_partitions = np.array_split(image_patches, num_partitions)
    label_patches_partitions = np.array_split(label_patches, num_partitions)
    return image_patches_partitions, label_patches_partitions, num_partitions

if Switch_splitDataset:

    print("########## LOAD DATASET")
    image_patches_BEN = np.load(os.path.join(config.DIR_TRAIN_BEN, "IMG", "BEN_IMG_PATCH.npy"))
    label_patches_BEN = np.load(os.path.join(config.DIR_TRAIN_BEN, "LBL", "BEN_LBL_PATCH.npy"))
    image_patches_CIV = np.load(os.path.join(config.DIR_TRAIN_CIV, "IMG", "CIV_IMG_PATCH.npy"))
    label_patches_CIV = np.load(os.path.join(config.DIR_TRAIN_CIV, "LBL", "CIV_LBL_PATCH.npy"))

    image_patches = np.concatenate((image_patches_BEN, image_patches_CIV), axis=0)
    label_patches = np.concatenate((label_patches_BEN, label_patches_CIV), axis=0)

    print("########## SPLIT DATASET TO TRAIN AND VAL")
    # SHUFFLE IMAGE PATCHES AND LABEL IDS
    indices = list(range(len(image_patches)))
    random.seed(16)
    random.shuffle(indices)
    image_patches_shuffled = [image_patches[i] for i in indices]
    label_patches_shuffled = [label_patches[i] for i in indices]
    threshold_train_val = int(len(image_patches_shuffled) * 0.7)

    image_patches_train, label_patches_train = image_patches_shuffled[:threshold_train_val], label_patches_shuffled[:threshold_train_val]
    image_patches_val, label_patches_val = image_patches_shuffled[threshold_train_val:], label_patches_shuffled[threshold_train_val:]

    np.save(os.path.join("/scratch.global/yin00406/Cashew_WA/TRAIN", "image_patches_src_train"), image_patches_train)
    np.save(os.path.join("/scratch.global/yin00406/Cashew_WA/TRAIN", "label_patches_src_train"), label_patches_train)
    np.save(os.path.join("/scratch.global/yin00406/Cashew_WA/TRAIN", "image_patches_src_val"), image_patches_val)
    np.save(os.path.join("/scratch.global/yin00406/Cashew_WA/TRAIN", "label_patches_src_val"), label_patches_val)

else:
    print("Load train & val dataset")

image_patches_train = np.load(os.path.join("/scratch.global/yin00406/Cashew_WA/TRAIN", "image_patches_src_train.npy"))
label_patches_train = np.load(os.path.join("/scratch.global/yin00406/Cashew_WA/TRAIN", "label_patches_src_train.npy"))
image_patches_val = np.load(os.path.join("/scratch.global/yin00406/Cashew_WA/TRAIN", "image_patches_src_val.npy"))
label_patches_val = np.load(os.path.join("/scratch.global/yin00406/Cashew_WA/TRAIN", "label_patches_src_val.npy"))
print(len(image_patches_train), len(image_patches_val))

print("BUILD MODEL")
model = model.STC_BN_TACA(in_channels=config.BAND_NUM, out_channels=config.CLS_NUM, time_steps=5)
model = model.to('cuda')
weights = torch.tensor([5.0, 10.0, 0.0, 2.0]).to("cuda")
criterion = torch.nn.CrossEntropyLoss(weight = weights, ignore_index=10, reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("TRAIN MODEL")
# Initial data in a dictionary format
data = {
    "epoch": [],
    "train_loss": [],
    "train_score_0": [],
    "train_score_1": [],
    "train_score_2": [],
    "max_score_0": [],
    "max_score_1": [],
    "max_score_2": [],

    # validation
    "vali_loss": [],
    "vali_score_0": [],
    "vali_score_1": [],
    "vali_score_2": [],
    "max_vali_score_0": [],
    "max_vali_score_1": [],
    "max_vali_score_2": []
}

df = pd.DataFrame(data)
df.to_csv(f'train_STC_BN_TACA_{learning_rate}.csv', index=False)

train_loss = []
train_score_0 = []
train_score_1 = []
train_score_2 = []
max_score_0 = 0
max_score_1 = 0
max_score_2 = 0

vali_loss = []
vali_score_0 = []
vali_score_1 = []
vali_score_2 = []
max_vali_score_0 = 0
max_vali_score_1 = 0
max_vali_score_2 = 0

for epoch in range(1, config.n_epochs + 1):
    # LOSS ON TRAIN SET
    model.train()
    epoch_loss = 0
    total_len = 0
    print("Epoch:{}".format(epoch))

    image_patches_partitions, label_patches_partitions, num_partitions = partitions(image_patches=image_patches_train,
                                                                                    label_patches=label_patches_train,
                                                                                    partition_size=partition_size,
                                                                                    rand=True)
    batch_num = 0

    for image_patches, label_patches in zip(image_patches_partitions, label_patches_partitions):
        data = utils.dataset(image_patches, label_patches)
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True,
                                                  num_workers=0, generator=torch.Generator(device='cuda'))
        if not 'pred_labels' in globals():
            pred_labels = np.zeros((num_partitions * partition_size * batch_size,
                                    patchSize - patchSize//4, patchSize - patchSize//4), dtype=np.int8)
            true_labels = np.zeros((num_partitions * partition_size * batch_size,
                                    patchSize - patchSize // 4, patchSize - patchSize // 4), dtype=np.int8)

        for batch, [image_batch, label_batch] in enumerate(data_loader):
            optimizer.zero_grad()

            out = model(image_batch.to('cuda'))
            out = out[:, :, patchSize//8:-patchSize//8, patchSize//8:-patchSize//8]
            label_batch = label_batch.type(torch.long).to('cuda')
            batch_loss = criterion(out, label_batch)
            batch_loss = torch.masked_select(batch_loss, (label_batch != 10))
            label_left = torch.masked_select(label_batch, (label_batch != 10))
            batch_loss = batch_loss.sum() / weights[label_left].sum()

            out = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
            pred_labels[total_len:total_len + len(image_batch)] = out.detach().cpu().numpy()
            true_labels[total_len:total_len + len(image_batch)] = label_batch.cpu().numpy() 
            total_len += len(image_batch)

            # LOSS BACKPROPOGATE
            batch_loss.backward()
            optimizer.step()

            # AGGREGATE LOSS
            epoch_loss += batch_loss.item()
            batch_num += 1

    epoch_loss = epoch_loss / batch_num
    print('\tTrain loss:{:.4f}'.format(epoch_loss), end="\t")
    train_loss.append(epoch_loss)

    pred_labels = pred_labels[:total_len]
    true_labels = true_labels[:total_len]
    pred_labels = np.reshape(pred_labels, (-1))
    true_labels = np.reshape(true_labels, (-1))
    pred_labels = pred_labels[true_labels != 10]
    true_labels = true_labels[true_labels != 10]
    f1_score_array = f1_score(y_true=true_labels, y_pred=pred_labels, average=None)
    train_score_0.append(f1_score_array[0])
    train_score_1.append(f1_score_array[1])
    train_score_2.append(f1_score_array[3])
    print("Score 0:{:.4f}\tScore 1:{:.4f}\tScore 2:{:.4f}\tMax Score 0:{:.4f}\tMax Score 1:{:.4f}\tMax Score 2:{:.4f}"
          .format(train_score_0[-1], train_score_1[-1], train_score_2[-1], max_score_0, max_score_1, max_score_2))

    if (max_score_0 < train_score_0[-1]) & (max_score_1 < train_score_1[-1]) & (max_score_2 < train_score_2[-1]):
        max_score_0 = train_score_0[-1]
        max_score_1 = train_score_1[-1]
        max_score_2 = train_score_2[-1]
        # torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "{}_{}.pt".format("STCA", epoch)))
    del pred_labels
    del true_labels

    model.eval()
    epoch_loss = 0
    total_len = 0

    image_patches_partitions, label_patches_partitions, num_partitions = partitions(image_patches=image_patches_val,
                                                                                    label_patches=label_patches_val,
                                                                                    partition_size=partition_size,
                                                                                    rand=True)
    batch_num = 0

    for image_patches, label_patches in zip(image_patches_partitions, label_patches_partitions):
        data = utils.dataset(image_patches, label_patches)
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True,
                                                  num_workers=0, generator=torch.Generator(device='cuda'))
        if not 'pred_labels' in globals():
            pred_labels = np.zeros((num_partitions * partition_size * batch_size,
                                    patchSize - patchSize // 4, patchSize - patchSize // 4), dtype=np.int8)
            true_labels = np.zeros((num_partitions * partition_size * batch_size,
                                    patchSize - patchSize // 4, patchSize - patchSize // 4), dtype=np.int8)

        for batch, [image_batch, label_batch] in enumerate(data_loader):

            out = model(image_batch.to('cuda'))
            out = out[:, :, patchSize // 8:-patchSize // 8, patchSize // 8:-patchSize // 8]
            label_batch = label_batch.type(torch.long).to('cuda')
            batch_loss = criterion(out, label_batch)
            batch_loss = torch.masked_select(batch_loss, (label_batch != 10))
            label_left = torch.masked_select(label_batch, (label_batch != 10))
            batch_loss = batch_loss.sum() / weights[label_left].sum()

            out = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
            pred_labels[total_len:total_len + len(image_batch)] = out.detach().cpu().numpy()
            true_labels[total_len:total_len + len(image_batch)] = label_batch.cpu().numpy()
            total_len += len(image_batch)

            # AGGREGATE LOSS
            epoch_loss += batch_loss.item()
            batch_num += 1

    epoch_loss = epoch_loss / batch_num
    print('\tVali loss:{:.4f}'.format(epoch_loss), end="\t")
    vali_loss.append(epoch_loss)

    pred_labels = pred_labels[:total_len]
    true_labels = true_labels[:total_len]
    pred_labels = np.reshape(pred_labels, (-1))
    true_labels = np.reshape(true_labels, (-1))
    pred_labels = pred_labels[true_labels != 10]
    true_labels = true_labels[true_labels != 10]
    f1_score_array = f1_score(y_true=true_labels, y_pred=pred_labels, average=None)
    vali_score_0.append(f1_score_array[0])
    vali_score_1.append(f1_score_array[1])
    vali_score_2.append(f1_score_array[3])
    print(
        "Vali Score 0:{:.4f}\tScore 1:{:.4f}\tScore 2:{:.4f}\tMax Score 0:{:.4f}\tMax Score 1:{:.4f}\tMax Score 2:{:.4f}"
        .format(vali_score_0[-1], vali_score_1[-1], vali_score_2[-1], max_vali_score_0, max_vali_score_1,
                max_vali_score_2))
    data_append = {
        "epoch": [epoch],
        "train_loss": [train_loss[-1]],
        "train_score_0": [train_score_0[-1]],
        "train_score_1": [train_score_1[-1]],
        "train_score_2": [train_score_2[-1]],
        "max_score_0": [max_score_0],
        "max_score_1": [max_score_1],
        "max_score_2": [max_score_2],

        # vali
        "vali_loss": [vali_loss[-1]],
        "vali_score_0": [vali_score_0[-1]],
        "vali_score_1": [vali_score_1[-1]],
        "vali_score_2": [vali_score_2[-1]],
        "max_vali_score_0": [max_vali_score_0],
        "max_vali_score_1": [max_vali_score_1],
        "max_vali_score_2": [max_vali_score_2]
    }

    df_append = pd.DataFrame(data_append)
    df_append.to_csv(f'train_STC_BN_TACA_{learning_rate}.csv', mode='a', header=False, index=False)

    if (max_vali_score_0 < vali_score_0[-1]) & (max_vali_score_1 < vali_score_1[-1]) & (
            max_vali_score_2 < vali_score_2[-1]):
        max_vali_score_0 = vali_score_0[-1]
        max_vali_score_1 = vali_score_1[-1]
        max_vali_score_2 = vali_score_2[-1]
        torch.save(model.state_dict(), os.path.join("/home/jinzn/yin00406/Cashew_mapping_WA/072024/TRAIN/trainedModel"
                                                    , "{}_{}_{}.pt".format("STC_BN_TACA", learning_rate, epoch)))
    del pred_labels
    del true_labels