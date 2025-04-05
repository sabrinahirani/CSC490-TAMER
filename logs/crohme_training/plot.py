import pandas as pd
import matplotlib.pyplot as plt

# Load the metrics.csv
log_path = "./logs/crohme_training/version_1/metrics.csv"
df = pd.read_csv(log_path)

# Fill missing values forward (useful for steps with missing logs)
df.fillna(method="ffill", inplace=True)

# Calculate the rescaling factor (scaling steps to 400 epochs)
max_steps = df['step'].max()
target_epochs = 400
scaling_factor = target_epochs / max_steps

# Rescale the 'step' to 'epoch'
df['epoch'] = df['step'] * scaling_factor

# Filter out training and validation losses
train_loss = df[df['train_loss'].notna()][['epoch', 'train_loss']]
val_loss = df[df['val_loss'].notna()][['epoch', 'val_loss']]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_loss['epoch'], train_loss['train_loss'], label='Train Loss')
plt.plot(val_loss['epoch'], val_loss['val_loss'], label='Val Loss')

# Set the x-axis to range from 0 to 400 epochs
plt.xlim(0, 400)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Scaled to 400 Epochs)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
print("Saved plot to loss_plot.png")
