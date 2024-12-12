import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def get_class_dictionary(image_sets_dir):   # name2idx
    classes = []

    # Lấy tất cả các file .txt có tên kết thúc bằng _train.txt
    class_files = [f for f in os.listdir(image_sets_dir) if f.endswith('_train.txt')]

    # Trích xuất tên class từ tên file (bỏ đuôi _train.txt)
    for class_file in sorted(class_files):
        class_name = class_file.replace('_train.txt', '')
        classes.append(class_name)

    # Loại bỏ các file trùng lặp và sắp xếp
    classes = sorted(list(set(classes)))

    # Tạo dictionary với tên class là key và index là value
    class_dict = {class_name: index for index, class_name in enumerate(classes)}

    return class_dict

class PascalVOC2012Dataset(Dataset):
    def __init__(self, root_dir, split='train', S = 7, B = 2, C = 20, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.S = S
        self.B = B
        self.C = C

        # # Default transform if none provided
        # if transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.Resize((448, 448)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        # else:
        #     self.transform = transform

        # Assuming a typical Pascal VOC directory structure
        img_dir = os.path.join(self.root_dir, 'JPEGImages')
        ann_dir = os.path.join(self.root_dir, 'Annotations')
        imgsets_dir = os.path.join(self.root_dir, 'ImageSets', 'Main')
        split_file = os.path.join(self.root_dir, 'ImageSets', 'Main', f'{self.split}.txt')

        self.class_dict = get_class_dictionary(imgsets_dir)

        # Load image and annotation paths
        self.images = []
        self.annotations = []

        # Read image names from split file
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]

        for img_name in image_names:
            img_path = os.path.join(img_dir, f'{img_name}.jpg')
            ann_path = os.path.join(ann_dir, f'{img_name}.xml')

            if os.path.exists(img_path) and os.path.exists(ann_path):
                self.images.append(img_path)
                self.annotations.append(ann_path)
            else:
                print(f"Image file missing: {img_path}")
                print(f"Annotation file missing: {ann_path}")
                print(f"Skipping missing file(s): {img_name}")


    def __len__(self):
        return len(self.images)


    def _parse_annotation(self, annotation_path):
        label_matrix = torch.zeros(self.S, self.S, 5 * self.B + self.C)

        try:
            anno_info = ET.parse(annotation_path)
            root = anno_info.getroot()
        except ET.ParseError:
            print(f"Error parsing XML file: {annotation_path}")
            return label_matrix

        # Image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)


        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_idx = self.class_dict[class_name]

            # Bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Normalize coordinates
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # Determine grid cell
            grid_x = int(x_center * self.S)
            grid_y = int(y_center * self.S)

            # Relative coordinates within grid cell
            x_offset = (x_center * self.S) - grid_x
            y_offset = (y_center * self.S) - grid_y

            # Update label matrix
            if label_matrix[grid_y, grid_x, 4] == 0:  # First box
                label_matrix[grid_y, grid_x, 0:4] = torch.tensor([x_offset, y_offset, box_width, box_height])
                label_matrix[grid_y, grid_x, 4] = 1  # Confidence
                label_matrix[grid_y, grid_x, 5:5+self.C] = torch.zeros(self.C)
                label_matrix[grid_y, grid_x, 10 + class_idx] = 1
            elif label_matrix[grid_y, grid_x, 9] == 0:  # Second box
                label_matrix[grid_y, grid_x, 5:9] = torch.tensor([x_offset, y_offset, box_width, box_height])
                label_matrix[grid_y, grid_x, 9] = 1  # Confidence

        return label_matrix


    def __getitem__(self, index):
        img_path = self.images[index]
        ann_path = self.annotations[index]

        # Load image
        to_tensor = transforms.ToTensor()
        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        mean = np.array(mean_rgb, dtype=np.float32)
        original_image = Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(448, 448), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # assuming the model is pretrained with RGB images.
        img = (img - mean) / 255.0 # normalize from -1.0 to 1.0.
        img = to_tensor(img)
        image_tensor = img.unsqueeze(0).to('cuda')

        label_matrix = self._parse_annotation(ann_path)

        return original_image, image_tensor, label_matrix


def collate_fn(batch):
    """
    Hàm collate tùy chỉnh cho dataset YOLO
    
    Args:
    - batch: List các tuple (original_image, image_tensor, label_matrix) từ dataset
    
    Returns:
    - original_images: List các ảnh gốc
    - images: Tensor batch các ảnh đã transform
    - labels: Tensor batch các nhãn
    """
    # Tách images, tensors và labels từ batch
    original_images, images, labels = zip(*batch)
    
    # Chuyển thành tensor
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return original_images, images, labels