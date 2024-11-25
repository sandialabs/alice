import os, sys
import pickle
import imageio
import numpy as np

data_folder = '.'
num_classes = 1000

# Make validation and training directories.
os.mkdir(data_folder + f'/val')
os.mkdir(data_folder + f'/train')
for i in range(num_classes):
    os.mkdir(data_folder + f'/val/{i+1:04d}')
    os.mkdir(data_folder + f'/train/{i+1:04d}')

def get_batch(filename, width=64):
    with open(filename, 'rb') as file_handle:
        batch = pickle.load(file_handle)
    width = 64
    pixels = width * width
    images = batch['data']
    images = np.dstack((images[:, :pixels], images[:, pixels:2*pixels], images[:, 2*pixels:])).reshape((images.shape[0], width, width, 3))
    return images, batch['labels']

# load validation data, process, and store
valid_file = data_folder + '/val_data'
val_images, val_labels = get_batch(valid_file)

# Save validation images using enumeration
count = 0
print('Validation: ')
for i, image in enumerate(val_images):
    count += 1
    imageio.imwrite(data_folder + f'/val/{val_labels[i]:04d}/{count:07d}.png', image)
    if (i+1) % 100 == 0:
        print('.', end='')
print('done.')

# Save training images continuing enumeration
for i in range(1, 11):
    print(f'Training {i}: ')
    train_file = data_folder + f'/train_data_batch_{i}'
    train_images, train_labels = get_batch(train_file)
    for j, image in enumerate(train_images):
        count += 1
        imageio.imwrite(data_folder + f'/train/{train_labels[j]:04d}/{count:07d}.png', image)
        if (j+1) % 100 == 0:
            print('.', end='')
    print('done.')

