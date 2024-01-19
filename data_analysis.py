import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def plot_mean_var_curve(time_bins, mean_curve, min_curve, max_curve, curve_name):
    plt.plot(time_bins, mean_curve, alpha=0.75, linewidth=.5, label=curve_name)
    plt.fill_between(time_bins, min_curve, max_curve, alpha=0.1)

def plt_init_fig():
    plt.figure(figsize=(15, 11))

def plt_save_fig(curve_save_path):
    plt.xlabel('Wallclock Time (relative)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(curve_save_path)


def wallclock_curve(path):
    """
    Given a psnr-to-wallclock curve extracted from w&b 
    that contains runs for different photos (DataFrame obj),
    return the average curve with shaded span (DataFrame obj).
    """
    df = pd.read_csv(path)

    start_time, end_time = get_start_end_time(df)
    time_bins = split_timespan(start_time, end_time)

    mean_curve = []
    min_curve = []
    max_curve = []
    for i in tqdm(range(len(time_bins) - 1)):
        point_sum = summary_at_bin(df, time_bins[i], time_bins[i+1])
        mean_curve.append(point_sum[0])
        min_curve.append(point_sum[1])
        max_curve.append(point_sum[2])
    
    time_bins = time_bins[:-1]
    mean_curve = np.array(mean_curve)
    min_curve = np.array(min_curve)
    max_curve = np.array(max_curve)

    return time_bins, mean_curve, min_curve, max_curve


def average_endtime(df):
    last_df = df.loc[df.filter(regex="_step$").eq(4999).any(axis=1)]
    return last_df['Relative Time (Process)'].mean()


def get_start_end_time(df):
    start_time = df['Relative Time (Process)'].iloc[0]
    end_time = average_endtime(df)
    
    return start_time, end_time


def split_timespan(start_time, end_time):
    return np.linspace(start_time, end_time, 5000)


def summary_at_bin(df, start_time, end_time):
    """Mean, min, max"""
    filtered_rows = df[(df["Relative Time (Process)"] >= start_time) & (df["Relative Time (Process)"] <= end_time)]

    # Calculate the mean of the specified column for the filtered rows
    mean_val = filtered_rows.filter(regex="psnr$").mean()[0]
    min_val = filtered_rows.filter(regex="psnr$").min()[0]
    max_val = filtered_rows.filter(regex="psnr$").max()[0]

    return (mean_val, min_val, max_val)


siren_100_csv_path = "csv/wandb_wallclock_psnr_siren100.csv"
siren_100_fig_path = "csv/wandb_wallclock_psnr_siren100.png"

if __name__ == "__main__":
    model = 'mlp'

    plt_init_fig()
    for mt in [20, 40, 60, 80, 100]:
        curve_name = "%s%s" % (model, mt)
        csv_path = "csv/wandb_wallclock_psnr_%s.csv" % curve_name
        time_bins, mean_curve, min_curve, max_curve = wallclock_curve(csv_path)
        plot_mean_var_curve(time_bins, mean_curve, min_curve, max_curve, curve_name)

    fig_path = "csv/wandb_wallclock_psnr_%s.png" % model
    plt_save_fig(fig_path)

