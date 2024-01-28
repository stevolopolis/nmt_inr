import pickle
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor


def loss_vid(losses, output_file, bins=40, interval=100):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        key, values = frame
        ax.hist(values, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(f'Histogram for step {key}')
        ax.set_xlabel('Loss')
        ax.set_ylabel('Frequency')

    frames = list(losses.items())

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)

    ani.save(output_file, writer='ffmpeg', fps=3)
    plt.close()


def sampling_iou(sample1, sample2, H, W):
    n_total = len(sample1)
    sample1_np = np.array(sample1)
    sample1_set = set([tuple(elem) for elem in sample1_np])
    sample2_np = np.array(sample2)
    sample2_set = set([tuple(elem) for elem in sample2_np])

    n_overlap = len(set.intersection(sample1_set, sample2_set))
    
    return n_overlap / n_total


def sampling_iou_curve(samples, H, W):
    iou_arr = []
    for i in range(len(samples) - 1):
        iou = sampling_iou(samples[i], samples[i+1], H, W)
        iou_arr.append(iou)

    return np.array(iou_arr) 


def plot_iou_curve(iou_arr, save_path):
    plt.plot(np.arange(len(iou_arr)), iou_arr)
    plt.xlabel('Fixed interval steps')
    plt.ylabel('IOU')

    plt.savefig(save_path)
    plt.close()


def normalize_samples(samples, H, W):
    """[-1, 1] to [H, W]"""
    samples = torch.tensor(samples)
    samples = (samples + 1) / 2
    samples = samples * torch.tensor([H, W])
    samples = samples.int().clamp(torch.tensor([0, 0]), torch.tensor([H-1, W-1]))

    return samples


def init_data(H, W, data_path=None):
    if data_path is None:
        return torch.ones((H, W, 3))
    else:
        img = Image.open(data_path)
        return ToTensor()(img)


def superimpose_samples(data, samples, data_range=1):
    if data_range == 1:
        vis_label = torch.tensor([0.75, 0.75, 0.75]).to(data.device)
    else:
        vis_label = torch.tensor([0.5, 0.0, 0.0]).to(data.device)

    data[samples[:, 0].long(), samples[:, 1].long(), :] = torch.clamp(data[samples[:, 0].long(), samples[:, 1].long(), :] - vis_label, max=1.0)

    return data.numpy()


def init_vid_saver(output_path, H, W, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    return out


def append_frame_to_vid(frame, out, last=False):
    frame = frame * 255
    frame = frame[:, :, ::-1]
    frame = frame.astype(np.uint8)
    out.write(frame)


def mt_vid(sample_history, output_path, H, W):
    vid_out = init_vid_saver(output_path, H, W, fps=10)
    for step in tqdm(sample_history.keys()):
        data = init_data(H, W, None)
        samples = normalize_samples(sample_history[step], H, W)
        frame = superimpose_samples(data, samples, data_range)
        append_frame_to_vid(frame, vid_out)
        
    vid_out.release()


if __name__ == '__main__':
    model = "siren"
    models = ['siren']  # ['mlp', 'siren', 'ffn']
    data_idx = '07'
    data_idxs = ['05'] # ['07', '14', '15', '17'] # , '05', '18', '20', '21'
    scheduler = 'constant'
    lr_scheduler = 'constant'
    mt_ratio = ['0.2', '0.4', '0.6', '0.8']

    for model in models:
        for data_idx in data_idxs:
            for ratio in mt_ratio:
                model_name = "%s_mt%s_%s_%s_kodak%s" % (model, ratio, scheduler, lr_scheduler, data_idx)
                sample_history_path = "logs/sampling/sampling_%s_2.pkl" % model_name
                loss_history_path = "logs/loss/loss_%s_2.pkl" % model_name
                data_path = "../datasets/kodak/kodim%s.png" % data_idx
                sampling_vid_path = "vis/dynamics/loss/sampling_%s_dynamics.mp4" % model_name
                loss_vid_path = "vis/dynamics/loss_%s_dynamics.mp4" % model_name
                iou_curve_path = "vis/iou/%s_iou_curve.png" % model_name
                H, W = 512, 768
                data_range = 1

                with open(sample_history_path, 'rb') as f:
                    sample_history = pickle.load(f)
                with open(loss_history_path, 'rb') as f:
                    loss_history = pickle.load(f)

                print("Converting sampling history to video...")
                mt_vid(sample_history, sampling_vid_path, H, W)
                print("Sampling history video completed.")

                print("Converting loss history to video...")
                loss_vid(loss_history, loss_vid_path)
                print("Loss history video completed.")

                print("Plotting IOU curve")

                sample_arr = []
                for step in tqdm(sample_history.keys()):
                    samples = normalize_samples(sample_history[step], H, W)
                    sample_arr.append(samples)

                iou_curve = sampling_iou_curve(sample_arr, H, W)
                plot_iou_curve(iou_curve, iou_curve_path)
                
                print("IOU curve completed.")
