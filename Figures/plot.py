import pandas as pd
import matplotlib.pyplot as plt
import os


# Load the data from the CSV file
ratios = [0.7, 0.55, 0.85]


# Ensure the directory for plots exists
plots_dir = './plots'
os.makedirs(plots_dir, exist_ok=True)

for ratio in ratios:
    file_path = f"./Finetune_loss/ratio_train={ratio}.csv"
    data = pd.read_csv(file_path)
    # Select data for the first 50 epochs
    plot_data = data.head(50)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['Epoch'], plot_data['Epoch Average Loss'], label='Training Loss')
    plt.plot(plot_data['Epoch'], plot_data['Validation Loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_file_path = os.path.join(plots_dir, f'train={ratio}_train_val_loss_plot.png')
    plt.savefig(plot_file_path)
    plt.show()
