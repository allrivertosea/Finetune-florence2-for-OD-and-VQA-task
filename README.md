# Environment
```shell
single RTX 3090
Ubuntu20.04
conda create --name florence python=3.11
conda activate florence
pip install -r requirements.txt
apt install git-lfs
```
# Dataset
Only DocumentVQA data, OD data is private, but you can download other opensource OD data from huggingface.
```shell
DocumentVQA:https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA
huggingface-cli download HuggingFaceM4/DocumentVQA --local-dir ./data
git clone https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA
git clone git@hf.co:datasets/HuggingFaceM4/DocumentVQA
```
# Model
```shell
https://huggingface.co/microsoft/Florence-2-base
huggingface-cli download microsoft/Florence-2-base-ft --local-dir ./microsoft
huggingface-cli download microsoft/Florence-2-large-ft --local-dir ./microsoft
```

# Object Detection Task
## finetune and visualization:
```shell
cd ./florence-2
python task_od/finetune.py
python task_od/lora.py
city-road-od_vis.ipynb
```
# Question Answer task
## finetune and visualization:
```shell
cd ./florence-2
python task_qa/finetune.py
python task_qa/lora.py
document_vqa_vis.ipynb
```
