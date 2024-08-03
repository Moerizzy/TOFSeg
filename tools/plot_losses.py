import pandas as pd
import matplotlib.pyplot as plt

# Load data from the CSV file
# df = pd.read_csv("lightning_logs/tof/dcswin/metrics.csv")
df = pd.read_csv("lightning_logs/tof/unetformer/metrics.csv")


# Define a function to calculate the running mean
def running_mean(series, window_size):
    return series.rolling(window=window_size, min_periods=1).mean()


# Add running mean calculations for 'train_loss_step' and 'val_loss'
df["train_loss_step_rm"] = running_mean(df["train_loss_step"], window_size=15)
df["val_loss_rm"] = running_mean(df["val_loss"], window_size=15)


# Function to plot the metrics
def plot_metrics(df, x, metrics):
    plt.figure(figsize=(15, 10))

    max_val_f1_epoch = df[df["val_F1"] == df["val_F1"].max()][x].values[0]

    for metric in metrics:
        data = df[[x, metric]].dropna()
        plt.plot(data[x], data[metric], label=metric)

    # Plot a red vertical line at the epoch with maximum val_F1
    plt.axvline(
        x=max_val_f1_epoch,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label="Model chosen",
    )

    print("max val_F1 epoch", max_val_f1_epoch)

    plt.xlabel(x)
    plt.ylabel("Value")
    # plt.title("Metrics Line Chart")
    plt.legend()
    plt.grid(True)
    plt.show()


# List of metrics to plot
metrics_to_plot = [
    # "train_loss_step",
    "train_loss_step_rm",
    # "epoch",
    "val_OA",
    "val_mIoU",
    # "val_IoU_Forest",
    "val_IoU_Tree",
    # "val_IoU_Background",
    "val_IoU_Linear",
    # "val_loss",
    # "val_IoU_Patch",
    "val_F1",
    # "train_IoU_Tree",
    # "train_OA",
    # "train_IoU_Linear",
    # "train_F1",
    # "train_loss_epoch",
    # "train_IoU_Background",
    # "train_IoU_Forest",
    # "train_mIoU",
    # "train_IoU_Patch",
]

# print max val_F1 and corresponding epoch
print("max val_F1 epoch", df.loc[df["val_F1"].idxmax()]["epoch"])
print("max val_F1", df["val_F1"].max())

# Convert columns to appropriate types (e.g., numeric)
df = df.apply(pd.to_numeric, errors="coerce")

# Plot the metrics
plot_metrics(df, "step", metrics_to_plot)
