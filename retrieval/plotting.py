import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_train_val_metric(train_values, val_values, metric_name, filename):
    plt.figure(figsize=(7, 5))
    plt.plot(train_values, label='train')
    plt.plot(val_values, label='val')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'Train and Validation {metric_name}')
    plt.grid(True)
    plt.savefig(filename)


def plot_action_counts(history, filename):
    df = pd.DataFrame(history).set_index('epoch')

    plt.figure(figsize=(7, 5))
    plt.plot(df['skip'], label='Skip', color='#7A8582', linewidth=2)
    plt.plot(df['take_1'], label='Take 1', color='#95bf74', linewidth=2)
    plt.plot(df['take_2p'], label='Take 2f', color='#659b5e', linewidth=2)
    plt.plot(df['take_2n'], label='Take 2b', color='#556f44', linewidth=2)
    plt.plot(df['take_3'], label='Take 3', color='#283f3b', linewidth=2)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Action Counts')
    plt.title('Action Selection per Epoch')
    plt.grid(True)
    plt.savefig(filename)
