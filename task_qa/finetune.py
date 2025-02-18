import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0" 

import argparse
import torch
import shutil
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from functools import partial
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor, get_scheduler)
import json
from data import DocVQADataset



def evaluate_model(model, val_loader, processor, epoch, device):
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
            inputs, answers = batch

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            val_loss += loss.item()
            # break

    avg_val_loss = val_loss / len(val_loader)
    print(f"Average Validation Loss: {avg_val_loss}")
        
def train_model(train_loader, 
                val_loader,
                model,
                processor,
                device,
                epochs=10, 
                lr=1e-6, 
                eval_steps=200,
                save_path="./qa_task_model_checkpoints",
                use_evaluate=False,
                ):
    
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        i = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            inputs, answers = batch

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            if i % eval_steps == 0 and use_evaluate:
                evaluate_model(model, val_loader, processor, epoch, device)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            # break

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")
        
        # Save model checkpoint
        model.eval()
        output_dir = os.path.join(save_path,"epoch_{}".format(epoch+1))
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config["vision_config"]["model_type"] = "davit"
    
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Validation phase
        if use_evaluate:
            evaluate_model(model, val_loader, processor, epoch, device)

def collate_fn(batch, processor,device):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs, answers
    
def main():
    parser = argparse.ArgumentParser(description="Train Florence-2 model on specified dataset")
    parser.add_argument("--dataset_name", default="docvqa", type=str, choices=["docvqa"], help="Dataset to train on")
    parser.add_argument("--data_path", default="/APP/florence-2/data/DocumentVQA", type=str, help="Dataset path")
    parser.add_argument("--model_path", default="/APP/florence-2/microsoft/Florence-2-base-ft", type=str, help="Model path")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--use_evaluate", type=bool, default=False, help="Use evaluate if this flag is passed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--save_path", type=str, default="./outputs_qa/test_qa_task_model_checkpoints", help="model save path")
    args = parser.parse_args()
    

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    print(torch.cuda.empty_cache())
    
    if args.dataset_name == "docvqa":
        train_dataset = DocVQADataset(split="train", data_path=args.data_path)
        val_dataset = DocVQADataset(split="validation", data_path=args.data_path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, processor=processor, device=device), num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, processor=processor, device=device), num_workers=0)
    
    for param in model.vision_tower.parameters():
        param.is_trainable = False
        
    train_model(train_loader, 
                val_loader,
                model,
                processor,
                device,
                epochs=args.epochs, 
                lr=args.lr, 
                eval_steps=args.eval_steps, 
                save_path=args.save_path,
                use_evaluate=args.use_evaluate
                )
    
if __name__ == "__main__":
    
    main()
