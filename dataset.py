import os
import math
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# import pytorch_lightning as pl
from einops import rearrange
from PIL import Image
import numpy as np
import random

# import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import sys


class ObjaverseDataLoader:
    def __init__(self, root_dir, batch_size, total_view=12, num_workers=4):
        # super().__init__(self, root_dir, batch_size, total_view, num_workers)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        image_transforms = [
            torchvision.transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ObjaverseData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=False,
            image_transforms=self.image_transforms,
        )
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        # sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=True,
            image_transforms=self.image_transforms,
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class ObjaverseData(Dataset):
    def __init__(
        self,
        root_dir=".objaverse/hf-objaverse-v1/views",
        image_transforms=None,
        total_view=12,
        validation=False,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.total_view = total_view

        # todo only partial data currently downloaded
        if os.path.exists(os.path.join(self.root_dir, "valid_paths.json")):
            with open(os.path.join(self.root_dir, "valid_paths.json")) as f:
                self.paths = json.load(f)
        else:
            self.paths = []
            # include all folders
            for folder in os.listdir(self.root_dir):
                if os.path.isdir(os.path.join(self.root_dir, folder)):
                    self.paths.append(folder)

        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[
                math.floor(total_objects / 100.0 * 99.0) :
            ]  # used last 1% as validation
        else:
            self.paths = self.paths[
                : math.floor(total_objects / 100.0 * 99.0)
            ]  # used first 99% as training
        print("============= length of dataset %d =============" % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(
            np.sqrt(xy), xyz[:, 2]
        )  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(
            T_target[None, :]
        )

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor(
            [
                d_theta.item(),
                math.sin(d_azimuth.item()),
                math.cos(d_azimuth.item()),
                d_z.item(),
            ]
        )
        return d_T

    def load_im(self, path, color):
        """
        replace background pixel with random color in rendering
        """
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.0] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.0))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = 12
        index_target, index_cond = random.sample(
            range(total_view), 2
        )  # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        color = [1.0, 1.0, 1.0, 1.0]

        try:
            target_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_target), color)
            )
            cond_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_cond), color)
            )
            target_RT = np.load(os.path.join(filename, "%03d.npy" % index_target))
            cond_RT = np.load(os.path.join(filename, "%03d.npy" % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(
                self.root_dir, "0a0c6d3b5f58499db8d6d649ba8de189"
            )  # this one we know is valid
            target_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_target), color)
            )
            cond_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_cond), color)
            )
            target_RT = np.load(os.path.join(filename, "%03d.npy" % index_target))
            cond_RT = np.load(os.path.join(filename, "%03d.npy" % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class LidcidriDataLoader:
    def __init__(self, root_dir, batch_size, total_view=12, num_workers=4):
        # super().__init__(self, root_dir, batch_size, total_view, num_workers)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        image_transforms = [
            torchvision.transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = LidcidriData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=False,
            image_transforms=self.image_transforms,
        )
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        # sampler=sampler)

    def val_dataloader(self):
        dataset = LidcidriData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=True,
            image_transforms=self.image_transforms,
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class LidcidriData(Dataset):
    def __init__(
        self,
        root_dir="./data/lidcidri/table_removed/img_simple",
        image_transforms=None,
        total_view=12,
        validation=False,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.total_view = total_view

        # todo only partial data currently downloaded
        if os.path.exists(os.path.join(self.root_dir, "valid_paths.json")):
            with open(os.path.join(self.root_dir, "valid_paths.json")) as f:
                self.paths = json.load(f)
        else:
            self.paths = []
            # include all folders
            for folder in os.listdir(self.root_dir):
                if os.path.isdir(os.path.join(self.root_dir, folder)):
                    self.paths.append(folder)

        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[
                math.floor(total_objects / 100.0 * 99.0) :
            ]  # used last 1% as validation
        else:
            self.paths = self.paths[
                : math.floor(total_objects / 100.0 * 99.0)
            ]  # used first 99% as training
        self.tform = image_transforms

        self.target_list = [i for i in range(-90, 95, 5) if i != 0]

    def __len__(self):
        return len(self.paths)

    # def cartesian_to_spherical(self, xyz):
    #     ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    #     xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    #     z = np.sqrt(xy + xyz[:, 2] ** 2)
    #     theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    #     # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    #     azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    #     return np.array([theta, azimuth, z])

    # def get_T(self, target_RT, cond_RT):
    #     R, T = target_RT[:3, :3], target_RT[:, -1]
    #     T_target = -R.T @ T

    #     R, T = cond_RT[:3, :3], cond_RT[:, -1]
    #     T_cond = -R.T @ T

    #     theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
    #     theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

    #     d_theta = theta_target - theta_cond
    #     d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    #     d_z = z_target - z_cond

    #     d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    #     return d_T

    def load_im(self, path):
        """
        Load grayscale image and do nothing
        """
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        # img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img * 255.0))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = self.total_view

        # Sampling method 1 : sample two distinct numbers from [-90, 90](degrees) with step=5
        # index_target, index_cond = random.sample(range(-90,95,5), 2)

        # Sampling method 2 : use 0 degree image as condition and sample 1 image from [-90, 90](except 0) with step=5 as target.
        index_cond = 0
        index_target = random.choice(self.target_list)
        patient_path = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        # color = [1., 1., 1., 1.]

        target_im = self.process_im(
            self.load_im(os.path.join(patient_path, "%d.png" % index_target))
        )
        cond_im = self.process_im(
            self.load_im(os.path.join(patient_path, "%d.png" % index_cond))
        )
        # target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
        # cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        # data["T"] = self.get_T(target_RT, cond_RT)
        d_azimuth = np.deg2rad((index_target - index_cond) % 360)
        data["T"] = torch.tensor(
            [0, math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), 0]
        )

        return data

    def process_im(self, im):
        im = im.convert("RGB") if im.mode != "RGB" else im
        return self.tform(im)


class LidcidriHardFBDataLoader:
    def __init__(self, root_dir, batch_size, total_view=12, num_workers=4):
        # super().__init__(self, root_dir, batch_size, total_view, num_workers)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        image_transforms = [
            torchvision.transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = LidcidriData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=False,
            image_transforms=self.image_transforms,
        )
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        # sampler=sampler)

    def val_dataloader(self):
        dataset = LidcidriData(
            root_dir=self.root_dir,
            total_view=self.total_view,
            validation=True,
            image_transforms=self.image_transforms,
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class LidcidriHardFBData(Dataset):
    def __init__(
        self,
        root_dir="./data/lidcidri/table_removed/img_complex_fb",
        image_transforms=None,
        total_view=3000,
        CT_thickness="all",  # "all" or "thin" or "thick"
        orientation="all",  # "all" or "PA" or "AP"
        validation=False,
        debug=True,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.total_view = total_view if orientation == "all" else total_view / 2
        if debug:
            self.total_view = 2

        if os.path.exists(os.path.join(self.root_dir, "patients.json")):
            with open(os.path.join(self.root_dir, "patients.json")) as f:
                self.paths = json.load(f)
                if CT_thickness == "thin":
                    self.paths = self.paths["thin"]
                elif CT_thickness == "thick":
                    self.paths = self.paths["thick"]
                else:  # all
                    self.paths = self.paths["thin"] + self.paths["thick"]
        else:
            self.paths = []
            # include all folders
            for folder in os.listdir(self.root_dir):
                if os.path.isdir(os.path.join(self.root_dir, folder)):
                    self.paths.append(folder)

        # read camera views
        with open(os.path.join(self.root_dir, "camera_views.json")) as f:
            camera_views = json.load(f)
            if orientation != "all":
                self.views = np.array(
                    [
                        view["coordinate"]
                        for view in camera_views
                        if view["orientation"] == orientation
                    ]
                )
            else:
                self.views = np.array([view["coordinate"] for view in camera_views])

        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[
                math.floor(total_objects / 100.0 * 99.0) :
            ]  # used last 1% as validation
        else:
            self.paths = self.paths[
                : math.floor(total_objects / 100.0 * 99.0)
            ]  # used first 99% as training
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)

    # def cartesian_to_spherical(self, xyz):
    #     ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    #     xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    #     z = np.sqrt(xy + xyz[:, 2] ** 2)
    #     theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    #     # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    #     azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    #     return np.array([theta, azimuth, z])

    def get_T(self, target_theta_phi, cond_theta_phi) -> torch.Tensor:
        (
            theta_cond,
            azimuth_cond,
        ) = (
            cond_theta_phi[0],
            cond_theta_phi[1],
        )
        (
            theta_target,
            azimuth_target,
        ) = (
            target_theta_phi[0],
            target_theta_phi[1],
        )

        # d_theta = theta_target - theta_cond
        # d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)

        # use cond - traget, because we flipped cond and target in training (by mistake)
        d_theta = theta_cond - theta_target
        d_azimuth = (azimuth_cond - azimuth_target) % (2 * math.pi)
        d_T = torch.tensor(
            [d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), 0]
        )
        print("cond_theta_phi: ", cond_theta_phi)
        print("target_theta_phi: ", target_theta_phi)
        print("d_T: ", d_T)
        print("azimuth_target", azimuth_target)
        print("azimuth_cond", azimuth_cond)
        print("d_azimuth", d_azimuth)
        return d_T

    def load_im(self, path):
        """
        Load grayscale image and do nothing
        """
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img = Image.fromarray(np.uint8(img * 255.0))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = self.total_view  # 3000

        # Sampling method 1 : sample two distinct numbers from [0, total_view)
        # index_target, index_cond = random.sample(range(0,total_view), 2)

        # Sampling method 2 : use image 0 as condition and sample 1 image from [1, total_view) as target.
        index_cond = 0
        index_target = random.randint(1, total_view - 1)
        patient_path = os.path.join(self.root_dir, self.paths[index])

        target_im = self.process_im(
            self.load_im(os.path.join(patient_path, "%04d.png" % index_target))
        )
        cond_im = self.process_im(
            self.load_im(os.path.join(patient_path, "%04d.png" % index_cond))
        )

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(self.views[index_target], self.views[index_cond])
        data["pid"] = self.paths[index]
        data["target_view"] = index_target

        return data

    def process_im(self, im):
        im = im.convert("RGB") if im.mode != "RGB" else im
        return self.tform(im)


# main
if __name__ == "__main__":
    # test dataloader
    # dataloader = ObjaverseDataLoader(root_dir='/data/zero123/views_release', batch_size=2, num_workers=4)
    dataloader = LidcidriDataLoader(
        root_dir="/work/XRAYDIFF/xiechun/data/lidcidri/table_removed/img_complex_fb_256",
        batch_size=2,
        num_workers=4,
    )

    train_loader = dataloader.train_dataloader()
    # import pdb; pdb.set_trace()
    for i, data in enumerate(dataloader.train_dataloader()):
        print(data)
