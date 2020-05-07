"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
import openslide
import cv2
from PIL import ImageFilter, Image
try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def get_random_wsi_region(filename, downsample=2, patch_size=256):

    '''
    takes a wsi and returns a random patch of patch_size
    downsample must be a multiple of 2

    '''

    try:
        wsi = openslide.OpenSlide(filename.numpy())
    except:
        wsi = openslide.OpenSlide(filename)
    # check power is 20X
    if '20' == wsi.properties['openslide.objective-power']:
        power = 20
    else:
        power = 40
        downsample*=2

    def find_tissue_mask():
        thumbnail = wsi.get_thumbnail((2000,2000))
        thumbnail_blurred = np.array(thumbnail.filter(ImageFilter.GaussianBlur(radius=10)))
        ret2,mask = cv2.threshold(thumbnail_blurred[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask[mask==0] = 1
        mask[mask==255] = 0
        return mask

    mask = find_tissue_mask()

    l_dims = wsi.level_dimensions
    level = wsi.get_best_level_for_downsample(downsample + 0.1)
    level_dims = l_dims[level]
    level_downsample = wsi.level_downsamples[level]
    mask_scale = 2000./max(level_dims)

    scale_factor = int(round(downsample / level_downsample))

    while True:
        x_start = int(np.random.uniform(low=0, high=level_dims[0]-patch_size*scale_factor))
        y_start = int(np.random.uniform(low=0, high=level_dims[1]-patch_size*scale_factor))

        mask_x_start = int(x_start*mask_scale)
        mask_x_stop = int((x_start + patch_size*scale_factor)*mask_scale)
        mask_y_start = int(y_start*mask_scale)
        mask_y_stop = int((y_start + patch_size*scale_factor)*mask_scale)

        if np.sum(mask[mask_y_start:mask_y_stop, mask_x_start:mask_x_stop]) > 0:

            region = wsi.read_region((int(x_start*level_downsample),int(y_start*level_downsample)), level, (patch_size*scale_factor,patch_size*scale_factor))

            if scale_factor > 1:
                region = region.resize((patch_size, patch_size), resample=1)
            region = np.array(region)[:,:,:3]
            region = region/127.5 - 1.
            return region

def load_test_data(image_path, fine_size=256):
    img = get_random_wsi_region(image_path)
    img = img/127.5 - 1
    return img

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = get_random_wsi_region(image_path[0])
    img_B = get_random_wsi_region(image_path[1])

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
