import concurrent.futures
import io
import os

import pandas as pd
from datasets import get_dataset_config_names, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random


def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]

def xywh2xyxy(bbox):
    """
    change bbox to coco format
    :param bbox: [x, y, w, h]
    :return: [x1, y1, x2, y2]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] + bbox[0],
        bbox[3] + bbox[1],
    ]
    
class BoxQuantizer(object):
    def __init__(self, mode='floor', bins=(1000,1000)):
        self.mode = mode
        self.bins = bins

    def quantize(self, boxes, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            quantized_xmin = int(xmin / size_per_bin_w)
            quantized_ymin = int(ymin / size_per_bin_h)
            quantized_xmax = int(xmax / size_per_bin_w)
            quantized_ymax = int(ymax / size_per_bin_h)

        elif self.mode == 'round':
            raise NotImplementedError()
        else:
            raise ValueError('Incorrect quantization type.')
       
        return quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax
    
    
class ODDataset(Dataset):
    def __init__(self, data_paths, split="train", class_names=["car","truck","person","bicycle","cyclist","van","tricycle","bus"]):
        self.name = "city-road"
        
        # Load all parquet files specified by data_paths
        self.data = []
        for data_path in data_paths:
            dataset = load_dataset("parquet", data_files={split: data_path})[split]
            self.data.extend(dataset)  # Concatenate the data from all files
        
        self.task_prompt = "<OD>"
        self.class_names = {i: name for i, name in enumerate(class_names)}
        self.names_class = {name: i for i, name in enumerate(class_names)}
        self.box_quantizer = BoxQuantizer()
    
    def build_answers(self, gt_bboxes, gt_labels, size):
        gt_labels = [self.class_names[gt_label] for gt_label in gt_labels] 
        
        answer = ""
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            x1, y1, x2, y2 = self.box_quantizer.quantize(gt_bbox, size)
            answer += "{}<loc_{}><loc_{}><loc_{}><loc_{}>".format(gt_label, x1, y1, x2, y2)
        return answer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        image = example['image']
        
        objects = example['objects']
        gt_bboxes = objects['bbox']
        gt_bboxes = [xywh2xyxy(gt_bbox) for gt_bbox in gt_bboxes]
        gt_labels = objects['category']
        size = image.size
        answers = self.build_answers(gt_bboxes, gt_labels, size)
        
        question = self.task_prompt
        
        # Ensure the image is in RGB mode
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return question, answers, image
