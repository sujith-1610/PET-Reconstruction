import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as io

epsilon = 1e-8

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=64, r_resolution=64, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.need_LR = need_LR
        self.split = split
        self.path = self.get_paths_from_images(dataroot)

        # Ensure dataset length is within valid range
        self.dataset_len = len(self.path)
        if self.dataset_len == 0:
            raise ValueError(f"No .mat files found in {dataroot}")

        self.data_len = min(data_len if data_len > 0 else self.dataset_len, self.dataset_len)
        self.num = max(1, self.data_len // 128)  # Avoid division by zero
        self.len = 0

    def get_paths_from_images(self, dataroot):
        paths = sorted([os.path.join(root, file) for root, _, files in os.walk(dataroot) for file in files if file.endswith('.mat')])
        return paths

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.len >= 128:
            self.len = 0

        if index >= len(self.path):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.path)}")

        # Construct image path
        base_path = '_'.join(self.path[index].split('_')[:-1])
        image_path = f"{base_path}_{self.len}.mat"
        self.len += 1

        # Load image safely
        try:
            image = io.loadmat(image_path)['img']
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

        if image.shape[1] < 256:
            raise ValueError(f"Invalid image shape {image.shape} in {image_path}, expected at least 256 pixels in width")

        image_h = torch.tensor(image[:, 128:256, :], dtype=torch.float32)
        image_s = torch.tensor(image[:, 0:128, :], dtype=torch.float32)
        
        img_Lpsd = self.make_psd(image_s)
        img_Hpsd = self.make_psd(image_h)
        img_3d_l = self.make_l3D(image_s, base_path, self.len - 1)
        img_3d_h = self.make_h3D(image_h, base_path, self.len - 1)

        # Generate negative samples safely
        negative_hpet = []
        for _ in range(10):
            negative_index = random.randint(0, self.num - 1) if self.num > 1 else 0
            negative_path = self.path[negative_index * 128]
            j = int(np.random.normal(0, 2, 1)[0])
            t = max(0, min(127, self.len + j - 1))  # Ensure valid frame index
            negative_image_path = f"{'_'.join(negative_path.split('_')[:-1])}_{t}.mat"

            try:
                negative_image = io.loadmat(negative_image_path)['img']
                negative_hpet.append(torch.tensor(negative_image[:, 128:256, :], dtype=torch.float32))
            except Exception as e:
                print(f"Error loading negative sample {negative_image_path}: {e}")
                negative_hpet.append(torch.zeros_like(image_h))  # Placeholder tensor

        negative_hpet = torch.cat(negative_hpet, dim=0) if negative_hpet else torch.zeros((10, *image_h.shape))

        result = {
            'HR': image_h, 'SR': image_s, 'LP': img_Lpsd, 'HP': img_Hpsd, 'NHR': negative_hpet,
            'L3D': img_3d_l, 'H3D': img_3d_h, 'Index': index
        }
        if self.need_LR:
            result['LR'] = torch.tensor(image[:, 0:128, :], dtype=torch.float32)
        return result

    def make_psd(self, img):
        try:
            img_numpy = img.permute(1, 2, 0).cpu().numpy()
            img_numpy = img_numpy.reshape(128, 128)
            fft = np.fft.fft2(img_numpy)
            fshift = np.fft.fftshift(fft) + epsilon
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            if magnitude_spectrum.max() != magnitude_spectrum.min():
                magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
                return torch.tensor(magnitude_spectrum[np.newaxis, :], dtype=torch.float32)
            else:
                return torch.zeros_like(img)
        except Exception as e:
            print(f"Error computing PSD: {e}")
            return torch.zeros_like(img)

    def make_l3D(self, img, base_path, index):
        return self.get_3D_stack(img, base_path, index, slice_type="low")

    def make_h3D(self, img, base_path, index):
        return self.get_3D_stack(img, base_path, index, slice_type="high")

    def get_3D_stack(self, img, base_path, index, slice_type="low"):
        offsets = [-2, -1, 0, 1, 2]
        stacked_images = []
        slice_range = (0, 128) if slice_type == "low" else (128, 256)

        for offset in offsets:
            frame_index = max(0, min(127, index + offset))
            file_path = f"{base_path}_{frame_index}.mat"
            try:
                image = io.loadmat(file_path)['img']
                stacked_images.append(torch.tensor(image[:, slice_range[0]:slice_range[1], :], dtype=torch.float32))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                stacked_images.append(torch.zeros_like(img))  # Placeholder

        return torch.cat(stacked_images, dim=0)

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
    sample = dataset.__getitem__(3)
    if sample:
        print(f"Sample loaded successfully: {list(sample.keys())}")
