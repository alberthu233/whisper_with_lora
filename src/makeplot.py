import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_comparison_curves(log_dir, events_out, model_names, results_path):
    # Dictionary to store the data for each model
    data = {model_name: {} for model_name in model_names}

    # Load data from TensorBoard logs
    for model_name in model_names:
        log_path = os.path.join(log_dir, events_out)
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        

        data[model_name]['train_loss'] = np.array([e.value for e in ea.Scalars(f'{model_name}/train_loss')])
        data[model_name]['val_loss'] = np.array([e.value for e in ea.Scalars(f'{model_name}/val_loss')])
        data[model_name]['val_auc'] = np.array([e.value for e in ea.Scalars(f'{model_name}/val_auc')])
        data[model_name]['train_time'] = np.cumsum([e.value for e in ea.Scalars(f'{model_name}/epoch_time')])

    # Create plots
    plt.figure()
    for model_name in model_names:
        epochs = range(1, len(data[model_name]['train_loss']) + 1)
        plt.plot(epochs, data[model_name]['train_loss'], label=f'{model_name} Train Loss')
        plt.plot(epochs, data[model_name]['val_loss'], label=f'{model_name} Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss vs. Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'loss_vs_epoch.png'))

    plt.figure()
    for model_name in model_names:
        train_time = data[model_name]['train_time'] - data[model_name]['train_time'][0]
        print(data[model_name]['train_time'])
        plt.plot(train_time, data[model_name]['train_loss'], label=f'{model_name} Train Loss')
        plt.plot(train_time, data[model_name]['val_loss'], label=f'{model_name} Val Loss')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss vs. Training Time')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'loss_vs_time.png'))

    plt.figure()
    for model_name in model_names:
        epochs = range(1, len(data[model_name]['val_auc']) + 1)
        plt.plot(epochs, data[model_name]['val_auc'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation AUC')
    plt.legend()
    plt.title('Validation AUC vs. Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'val_auc_vs_epoch.png'))

    plt.figure()
    for model_name in model_names:
        train_time = data[model_name]['train_time'] - data[model_name]['train_time'][0]
        plt.plot(train_time, data[model_name]['val_auc'], label=model_name)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Validation AUC')
    plt.legend()
    plt.title('Validation AUC vs. Training Time')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'val_auc_vs_time.png'))

if __name__ == '__main__':
    log_dir = 'logs'  # Directory containing the TensorBoard logs
    model_names = ['full_finetune', 'lora']  # Names of the models to compare
    results_path = 'images'  # Directory to save the plots
    events_out = 'events.out.tfevents.1710471533.DSI-DGX01.227336.0'  # Name of the TensorBoard event file
    plot_comparison_curves(log_dir, events_out, model_names, results_path)