import concurrent.futures
import io

import pandas as pd
from datasets import get_dataset_config_names, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random

class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.name = "BaseDataset"
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


class DocVQADataset(BaseDataset):
    def __init__(self, split, data_path, use_all=False):
        super().__init__(split)
        self.name = "DocVQA"
        self.data = load_dataset(data_path, split=split)
        self.task_prompt = "<VQA>"
        self.split = split
        self.use_all = use_all

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.task_prompt + self.correct_casing_finqa(
            example["question"], True
        )
        answers = example["answers"]
        
        if not self.use_all:
            random.shuffle(answers)
            answers = answers[0]

        image = example["image"]  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # print("question:",question)
        # print("answers:",answers)
     
        return question, answers, image