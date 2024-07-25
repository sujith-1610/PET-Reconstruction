import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as io
from PIL import Image
from medpy.io import load
from torchvision import transforms

epsilon = 1e-8

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=64, r_resolution=64, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.path = self.get_paths_from_images(dataroot)
        self.dataset_len = len(self.path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        self.num = int(self.data_len / 128)
        self.len = 0

    def get_paths_from_images(self, dataroot):
        paths = []
        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.endswith('.mat'):
                    paths.append(os.path.join(root, file))
        return sorted(paths)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.len == 128:
            self.len = 0
        image_path = os.path.join(self.path[index])
        image_path = '_'.join(image_path.split('_')[:-1]) + '_' + str(self.len) + '.mat'
        self.len += 1

        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        img_hpet = torch.Tensor(image_h)
        image_s = image[:, 0:128, :]
        img_spet = torch.Tensor(image_s)
        if self.need_LR:
            image_l = image[:, 0:128, :]
            img_lpet = torch.Tensor(image_l)

        img_Lpsd = self.make_psd(img_spet)
        img_Hpsd = self.make_psd(img_hpet)
        img_3d_l = self.make_l3D(img_spet, self.path[index], self.len-1)
        img_3d_h = self.make_h3D(img_hpet, self.path[index], self.len-1)

        for i in range(10):
            if self.num > 1:
                negative_index = random.randint(0, self.num - 1)
                while negative_index == int(index / 128):
                    negative_index = random.randint(0, self.num - 1)
            else:
                print(f"self.num is less than or equal to 1: {self.num}")
                negative_index = 0

            negative_path = os.path.join(self.path[negative_index * 128])
            s = np.random.normal(0, 2, 1)
            j = int(s[0])
            t = self.len + j - 1
            if 0 <= t <= 127:
                negative_image_path = '_'.join(negative_path.split('_')[:-1]) + '_' + str(t) + '.mat'
            else:
                negative_image_path = '_'.join(negative_path.split('_')[:-1]) + '_' + str(self.len - 1) + '.mat'

            negative_image = io.loadmat(negative_image_path)['img']
            negative_image = negative_image[:, 128:256, :]
            negative_image = torch.Tensor(negative_image)
            if i == 0:
                negative_hpet = negative_image
            else:
                negative_hpet = torch.cat((negative_hpet, negative_image), 0)

        if self.need_LR:
            return {'LR': img_lpet, 'HR': img_hpet, 'SR': img_spet, 'NHR': negative_hpet,
                    'LP': img_Lpsd, 'HP': img_Hpsd, 'L3D': img_3d_l, 'H3D': img_3d_h,
                    'Index': index}
        else:
            return {'HR': img_hpet, 'SR': img_spet, 'LP': img_Lpsd, 'HP': img_Hpsd, 'NHR': negative_hpet,
                    'L3D': img_3d_l, 'H3D': img_3d_h, 'Index': index}

    def make_psd(self, img):
        gen_imgs = img.permute(1, 2, 0)
        img_numpy = gen_imgs[:, :, :].cpu().detach().numpy()
        img_numpy = np.reshape(img_numpy, (128, 128))
        fft = np.fft.fft2(img_numpy)
        fshift = np.fft.fftshift(fft)
        fshift += epsilon
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        if (magnitude_spectrum.max() - magnitude_spectrum.min()) != 0:
            magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
            magnitude_spectrum = magnitude_spectrum[np.newaxis, :].astype(np.float32)
        else:
            magnitude_spectrum = np.zeros_like(img)
        return magnitude_spectrum

    def make_l3D(self, img, path, index):
        if index - 2 < 0:
            result = img
        else:
            image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 2) + '.mat'
            image = io.loadmat(image_path)['img']
            image_h = image[:, 0:128, :]
            result = torch.Tensor(image_h)
        if index - 1 < 0:
            result = torch.cat((result, img), 0)
        else:
            image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 1) + '.mat'
            image = io.loadmat(image_path)['img']
            image_h = image[:, 0:128, :]
            new_result = torch.Tensor(image_h)
            result = torch.cat((result, new_result), 0)

        result = torch.cat((result, img), 0)

        if index + 1 > 127:
            result = torch.cat((result, img), 0)
        else:
            image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 1) + '.mat'
            image = io.loadmat(image_path)['img']
            image_h = image[:, 0:128, :]
            new_result = torch.Tensor(image_h)
            result = torch.cat((result, new_result), 0)

        if index + 2 > 127:
            result = torch.cat((result, img), 0)
        else:
            image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 2) + '.mat'
            image = io.loadmat(image_path)['img']
            image_h = image[:, 0:128, :]
            new_result = torch.Tensor(image_h)
            result = torch.cat((result, new_result), 0)

        return result

    def make_h3D(self, img, path, index):
        if index - 2 < 0:
            result = img
        else:
            image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 2) + '.mat'
            image = io.loadmat(image_path)['img']
            image_h = image[:, 128:256, :]
            result = torch.Tensor(image_h)
        if index - 1 < 0:
            result = torch.cat((result, img), 0)
        else:
            image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 1) + '.mat'
            image = io.loadmat(image_path)['img']
            image_h = image[:, 128:256, :]
            new_result = torch.Tensor(image_h)
            result = torch.cat((result, new_result), 0)

        result = torch.cat((result, img), 0)

        if index + 1 > 127:
            result = torch.cat((result, img), 0)
        else:
            image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 1) + '.mat'
            image = io.loadmat(image_path)['img']
            image_h = image[:, 128:256, :]
            new_result = torch.Tensor(image_h)
            result = torch.cat((result, new_result), 0)

        if index + 2 > 127:
            result = torch.cat((result, img), 0)
        else:
            image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 2) + '.mat'
            image = io.loadmat(image_path)['img']
            image_h = image[:, 128:256, :]
            new_result = torch.Tensor(image_h)
            result = torch.cat((result, new_result), 0)

        return result

if __name__ == '__main__':
    dataroot = 'E:/数据集/Desktop/train_mat'
    dataset = LRHRDataset(
        dataroot=dataroot,
        datatype='jpg',
        l_resolution=64,
        r_resolution=64,
        split='train',
        data_len=-1,
        need_LR=False
    )
    dataset.__getitem__(3)
