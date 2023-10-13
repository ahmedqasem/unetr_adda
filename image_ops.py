import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify
import nibabel as nib
import ipywidgets as widgets
import os


def apply_window(img, width=500, center=40):
    # np.interp(a, (a.min(), a.max()), (-1, +1))

    # convert below window to black
    img[img<(center-(width/2))]=center-(width/2)
    # convert above window to white
    img[img>(center+(width/2))]=center+(width/2)

    # normalize image
    img_scaled = np.interp(img, (img.min(), img.max()), (0, +1))
    # print(img_scaled.shape)
    # print(np.min(img_scaled), np.max(img_scaled))
    return img


# Define a function to visualize the data
def explore_3dimage(layer, ct_image_data):
    fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
    ax[0].imshow(apply_window(ct_image_data[:, :, layer]), cmap='gray')
    return layer


def explore_3dimage_interact(layer, ct_image_data):
    return widgets.interact(explore_3dimage, layer=layer, ct_image_data=ct_image_data)


def load_sample_image():
    root_images_folder = '../datasets/ADDA/resampled/images'
    root_labels_folder = '../datasets/ADDA/resampled/labels'
    img_name = 'MDA-126'

    ct_obj = nib.load(os.path.join(root_images_folder, f'{img_name}__CT.nii.gz'))
    ct_image_data = ct_obj.get_fdata()

    lbl_obj = nib.load(os.path.join(root_labels_folder, f'{img_name}.nii.gz'))
    lbl_image_data = ct_obj.get_fdata()

    # Get the image shape and print it out
    height, width, depth = ct_image_data.shape
    # print(f"The image object has the following dimensions: height: {height}, width:{width}, depth:{depth}")

    # patches
    ct_patches = patchify(ct_image_data, (256, 256, 256), step=128)
    mask_patches = patchify(lbl_image_data, (256, 256, 256), step=128)

    return ct_patches, mask_patches


if __name__ == '__main__':
    patches, labels = load_sample_image()
    print(patches.shape, labels.shape)