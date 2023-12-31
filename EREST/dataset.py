import os
import os.path
import torch

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def make_dataset(root):
    images = []
    frame = 3
    for target in sorted(os.listdir(root)):
        subfolder_path = os.path.join(root, target, 'img')
        item_frames = []
        numimg = len(os.listdir(subfolder_path))
        epo = numimg//(frame)
        all_name = sorted(os.listdir(subfolder_path))
        for mm in range(0, epo):
            one_name = all_name[mm*(frame):mm*(frame)+frame]
            i = 1
            for fi in one_name:
                _, ending = os.path.splitext(fi)
                if ending == ".jpg":
                    if i % 1 == 0:
                        img_file_name = fi
                        img_file_path = os.path.join(subfolder_path, img_file_name)  # eg: dir + '/Davis/bear/1.jpg'
                        item = (img_file_path, target)
                        item_frames.append(item)
                        if i % frame == 0 and i > 0:
                            images.append(item_frames)
                            item_frames = []
                            break
                    i = i+1
        if mm == (epo-1):
            one_name = all_name[numimg-(frame):numimg+2]
            i = 1
            for fi in one_name:
                _, ending = os.path.splitext(fi)
                if ending == ".jpg":
                    if i % 1 == 0:
                        img_file_name = fi
                        img_file_path = os.path.join(subfolder_path, img_file_name)  # eg: dir + '/Davis/bear/1.jpg'
                        item = (img_file_path, target)
                        item_frames.append(item)
                        if i % frame == 0 and i > 0:
                            images.append(item_frames)
                            item_frames = []
                            break
                    i = i+1
    return images


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        clip = self.imgs[index]
        img_clip = []
        img_name = []

        i = 0
        for frame in clip:
            img_path, name = frame
            imgname = img_path.split('\\')
            img = Image.open(img_path).convert('RGB')
            width = img.size[0]
            height = img.size[1]
            i = i+1
            if self.transform is not None:
                img = self.transform(img)

            img = img.view(1, img.size(0), img.size(1), img.size(2))
            img_clip.append(img)
            img_name.append(imgname[-1])
        img = torch.cat(img_clip, 0)

        return img, name, img_name, width, height

    def __len__(self):
        return len(self.imgs)
