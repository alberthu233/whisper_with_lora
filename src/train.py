import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import whisper

from dataset import LigoBinaryData, two_channel_LigoBinaryData
from model import BaselineModel, ligo_binary_classifier, two_channel_ligo_binary_classifier, LoRA_layer, LoRa_linear
from utils import EarlyStopper, save_plot, get_paths
import copy

def evaluate(model, data_loader, device, criterion):
    all_labels = []
    all_preds = []
    all_raw_preds = []
    all_snr = []
    total_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            input_0, input_1, labels, snr = batch

            input_0 = input_0.to(device)
            input_1 = input_1.to(device)
            
            logits = model(input_0, input_1)
            
            labels = labels.view(-1, 1).float().to(device)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).round()
            raw_preds = torch.sigmoid(logits)
            all_raw_preds.append(raw_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_snr.append(snr.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_raw_preds = np.concatenate(all_raw_preds, axis=0)
    all_snr = np.concatenate(all_snr, axis=0)
    
    loss = total_loss / len(data_loader)
    auc = roc_auc_score(all_labels, all_raw_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_raw_preds)
    report = classification_report(all_labels, all_preds, target_names=['injection', 'noise'])
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    eval_out = {
        'loss': loss,
        'auc': auc,
        'f1': f1,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_raw_preds': all_raw_preds,
        'report': report,
        'fpr': fpr,
        'tpr': tpr,
        'all_snr': all_snr
    }
    
    return eval_out

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, results_path, checkpoint_path, model_name, writer):
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=20)
    
    train_losses = []
    val_losses = []
    val_aucs = []
    train_times = []
    
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        
        train_loss = 0.0
        for input_0, input_1, labels, snr in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            optimizer.zero_grad()
            
            input_0 = input_0.to(device)
            input_1 = input_1.to(device)
            labels = labels.view(-1, 1).float().to(device)
            
            outputs = model(input_0, input_1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        train_times.append(epoch_time)
        
        eval_out = evaluate(model, val_loader, device, criterion)
        val_loss = eval_out['loss']
        val_auc = eval_out['auc']
        
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        writer.add_scalar(f'{model_name}/train_loss', train_loss, epoch)
        writer.add_scalar(f'{model_name}/val_loss', val_loss, epoch)
        writer.add_scalar(f'{model_name}/val_auc', val_auc, epoch)
        writer.add_scalar(f'{model_name}/epoch_time', epoch_time, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
        
        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses, val_aucs, train_times

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=args.log_dir)
    
    ds = load_from_disk(args.data_path)
    ds_split = ds.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    
    train_data = two_channel_LigoBinaryData(ds_split['train'], device)
    valid_data = two_channel_LigoBinaryData(ds_split['test'], device)
        
    whisper_model = whisper.load_model(args.encoder)
    encoder = whisper_model.encoder
    
    models = [
        ('lora', two_channel_ligo_binary_classifier(encoder)),
        ('frozen_encoder', two_channel_ligo_binary_classifier(encoder)),
        ('full_finetune', two_channel_ligo_binary_classifier(encoder))
    ]
    
    for model_name, model in models:
        model = copy.deepcopy(model)
        if model_name == 'frozen_encoder' or model_name == 'lora':
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

        elif model_name == 'full_finetune':
            for param in model.parameters():
                param.requires_grad = True
        
        if model_name == 'lora':
            replace_lora = partial(LoRa_linear, rank=args.lora_rank, alpha=args.lora_alpha)
            for layer in model.encoder.blocks:
                if args.lora_q:
                    layer.attn.query = replace_lora(layer.attn.query)
                if args.lora_k:
                    layer.attn.key = replace_lora(layer.attn.key)
                if args.lora_v:
                    layer.attn.value = replace_lora(layer.attn.value)
                if args.lora_mlp:
                    for i, mlp_layer in enumerate(layer.mlp):
                        if isinstance(mlp_layer, torch.nn.Linear):
                            layer.mlp[i] = replace_lora(mlp_layer)
        
        model.to(device)
        
        if model_name == 'lora':
            train_loader = DataLoader(train_data, batch_size=args.batch_size * 2, shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(valid_data, batch_size=args.batch_size * 2, shuffle=True, num_workers=args.num_workers)
        if model_name == 'frozen_encoder':
            train_loader = DataLoader(train_data, batch_size=args.batch_size * 2, shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(valid_data, batch_size=args.batch_size * 2, shuffle=True, num_workers=args.num_workers)   
        if model_name == 'full_finetune':
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Training {model_name} model with {params/1e6}M trainable parameters...")
        train_losses, val_losses, val_aucs, train_times = train(model, train_loader, valid_loader, optimizer, criterion, device, args.num_epochs, args.results_path, f'{model_name}_checkpoint.pt', model_name, writer)
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LoRA parameters configuration')
    parser.add_argument('--data_path', type=str, default='/workspace/ligo_general/data/O3b_train_real_noise_20k_20Hz_new_128_resampled_test', help='Path to the dataset')
    parser.add_argument('--results_path', type=str, default='/workspace/ligo_general/data/results', help='Path to save the results')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save the TensorBoard logs')
    parser.add_argument('--encoder', type=str, default='small', help='Encoder to use (small, base, or large)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of workers')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=8, help='LoRA alpha')
    parser.add_argument('--lora_q', action='store_true', help='LoRA q flag')
    parser.add_argument('--lora_k', action='store_true', help='LoRA k flag')
    parser.add_argument('--lora_v', action='store_true', help='LoRA v flag')
    parser.add_argument('--lora_mlp', action='store_true', help='LoRA mlp flag')
    args = parser.parse_args()
    
    main(args)