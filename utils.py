import numpy as np
from torch.utils.data.dataset import Dataset
from collections import Counter

def create_patches_train_sequential(img_array, label_array, patchSize, switch_labelValid=False):
    height, width = label_array.shape
    image_patches = []
    label_patches = []
    count = 0
    output_patch_size = patchSize - patchSize//4
    stride = patchSize // 2
    borderRemoved = patchSize//8
    for i in range(height // stride):
        for j in range(width // stride):

            i_label_start, i_label_end = i * stride+borderRemoved, i * stride + borderRemoved + output_patch_size
            i_image_start, i_image_end = i * stride, i * stride + patchSize
            j_label_start, j_label_end = j * stride+borderRemoved, j * stride + borderRemoved + output_patch_size
            j_image_start, j_image_end = j * stride, j * stride + patchSize
            if 0 <= i_image_start < height and 0 < i_image_end <= height and 0 <= j_image_start < width and 0 < j_image_end <= width:
                count += 1
                image_patch = img_array[:, :, i_image_start:i_image_end, j_image_start:j_image_end]
                label_patch = label_array[i_label_start:i_label_end, j_label_start:j_label_end]

                image_patches.append(image_patch)
                label_patches.append(label_patch)

    print(f"The number of clipped patches: {count}")
    return image_patches, label_patches

def create_patches_train_sequential_dispersed(img_array, label_array, label_array_valid, patchSize, threshold=0.5):
    height, width = label_array.shape
    image_patches = []
    label_patches = []
    count = 0
    output_patch_size = patchSize - patchSize//4
    stride = patchSize // 4
    borderRemoved = patchSize//8
    for i in range(height // stride):
        for j in range(width // stride):
            
            i_label_start, i_label_end = i * stride+borderRemoved, i * stride + borderRemoved + output_patch_size
            i_image_start, i_image_end = i * stride, i * stride + patchSize
            j_label_start, j_label_end = j * stride+borderRemoved, j * stride + borderRemoved + output_patch_size
            j_image_start, j_image_end = j * stride, j * stride + patchSize
            if 0 <= i_image_start < height and 0 < i_image_end <= height and 0 <= j_image_start < width and 0 < j_image_end <= width:

                image_patch = img_array[:, :, i_image_start:i_image_end, j_image_start:j_image_end]
                label_patch = label_array[i_label_start:i_label_end, j_label_start:j_label_end]
                label_valid_patch = label_array_valid[i_label_start:i_label_end, j_label_start:j_label_end]

                if np.sum(label_valid_patch) >= threshold * output_patch_size * output_patch_size:
                    count += 1
                    image_patches.append(image_patch)
                    label_patches.append(label_patch)

    print(f"The number of clipped patches: {count}")
    return image_patches, label_patches

class dataset(Dataset):

    def __init__(self, image_patches, label_patches, transform=None):
        self.image_patches = image_patches
        self.label_patches = label_patches
        self.transform = transform

    def __len__(self):
        return len(self.label_patches)

    def __getitem__(self, index):
        if self.transform:
            image_patch = self.transform(self.image_patches[index])

        return self.image_patches[index], self.label_patches[index]

def histogram(array):
    for key, value in sorted(Counter(np.reshape(array, (-1))).items()):
        print("{} : {}".format(key, value))
