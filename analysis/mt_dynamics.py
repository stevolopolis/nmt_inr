import pickle
import torch
import cv2
import os
import skimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

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
    #matplotlib.rcParams.update({'font.size': 18})
    plt.plot(np.arange(len(iou_arr)) * 100, iou_arr)
    plt.xlabel('Steps')
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


def init_data(H, W, data_path=None, resize=False):
    if data_path is None:
        return torch.ones((H, W, 3))
    elif data_path == "camera":
        img =  Image.fromarray(skimage.data.camera()).convert('RGB')
        return ToTensor()(img).moveaxis(0, -1)
    else:
        img = Image.open(data_path)
        img = img.resize((W, H)) if resize else img
        return ToTensor()(img).moveaxis(0, -1)


def superimpose_samples(data, samples, data_range=1):
    if data_range == 1:
        vis_label = torch.tensor([0.75, 0.0, 0.0]).to(data.device)
    else:
        vis_label = torch.tensor([0.5, 0.0, 0.0]).to(data.device)

    data[samples[:, 0].long(), samples[:, 1].long(), :] = vis_label

    return data.numpy()


def samples2frame(samples, H, W, data_range=1):
    frame = torch.zeros((H, W, 3), dtype=torch.float32)
    if data_range == 1:
        vis_label = torch.tensor([0.75, 0.75, 0.75])
    else:
        vis_label = torch.tensor([0.5, 0.5, 0.5])

    frame[samples[:, 0].long(), samples[:, 1].long(), :] = vis_label

    return frame.numpy()


def init_vid_saver(output_path, H, W, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    return out


def append_frame_to_vid(frame, out, last=False):
    frame = frame * 255
    frame = frame[:, :, ::-1]
    frame = frame.astype(np.uint8)
    out.write(frame)


def save_frame(frame, output_dir, step):
    frame = frame * 255
    frame = frame[:, :, ::-1]
    frame = frame.astype(np.uint8)
    cv2.imwrite("%s/%s.png" % (output_dir, step), frame)


def mt_vid(sample_history, data_path, output_path, output_dir, H, W, fps=10):
    vid_out = init_vid_saver(output_path, H, W, fps=fps)
    if os.path.basename(output_dir) not in os.listdir(os.path.dirname(output_dir)):
        os.mkdir(output_dir)
    for step in tqdm(sample_history.keys()):
        data = init_data(H, W, data_path=data_path)
        samples = normalize_samples(sample_history[step], H, W)
        frame = superimpose_samples(data, samples, data_range)
        append_frame_to_vid(frame, vid_out)
        save_frame(frame, output_dir, step)
        
    vid_out.release()


def recon_vid(wandb_log_path, output_path, output_dir, H, W, fps=10):
    if os.path.basename(output_dir) not in os.listdir(os.path.dirname(output_dir)):
        os.mkdir(output_dir)
    iterator = [0] + [i for i in range(49, 4999, 50)]
    for step in tqdm(iterator):
        # check for filename regex
        img_file = glob.glob(f"{wandb_log_path}/Reconstruction_{step}*.png")
        if len(img_file) == 0:
            print(f"File not found for step {step}")
            continue
        img_file = img_file[0]
        frame = init_data(H, W, data_path=img_file)
        if step == 0:    
            vid_out = init_vid_saver(output_path, frame.shape[0], frame.shape[1], fps=fps)
        frame = frame.numpy()
        append_frame_to_vid(frame, vid_out)
        save_frame(frame, output_dir, step)
        
    vid_out.release()


def sample_vid(sample_history, output_path, output_dir, H, W, fps=10):
    vid_out = init_vid_saver(output_path, H, W, fps=fps)
    if os.path.basename(output_dir) not in os.listdir(os.path.dirname(output_dir)):
        os.mkdir(output_dir)
    for step in tqdm(sample_history.keys()):
        samples = normalize_samples(sample_history[step], H, W)
        frame = samples2frame(samples, H, W, data_range=1)
        append_frame_to_vid(frame, vid_out)
        save_frame(frame, output_dir, step)
        
    vid_out.release()


def combined_vid(data_path, wandb_log_path, sample_history, output_path, output_dir, H, W, fps=10):
    if os.path.basename(output_dir) not in os.listdir(os.path.dirname(output_dir)):
        os.mkdir(output_dir)
    iterator = [0] + [i for i in range(49, 4999, 50)]
    for step in tqdm(iterator):
        # load reconstruction
        img_file = glob.glob(f"{wandb_log_path}/Reconstruction_{step}*.png")
        if len(img_file) == 0:
            print(f"File not found for step {step}")
            continue
        img_file = img_file[0]
        recon_frame = init_data(H, W, data_path=img_file)
        if step == 0:    
            vid_out = init_vid_saver(output_path, recon_frame.shape[0]*3, recon_frame.shape[1], fps=fps)
        recon_frame = recon_frame.numpy()
        # load gt data
        gt_frame = init_data(recon_frame.shape[0], recon_frame.shape[1], data_path=data_path, resize=True)
        # load samples
        if step == 0:
            sample_key = str(step)
        else:
            sample_key = str(step+1)
        # check if sample key exists
        if sample_key not in sample_history.keys():
            print(f"Sample key {sample_key} not found.")
            continue
        samples = normalize_samples(sample_history[sample_key], recon_frame.shape[0], recon_frame.shape[1])
        sample_frame = samples2frame(samples, recon_frame.shape[0], recon_frame.shape[1], data_range=1)
        # combine frames
        combined_frame = np.concatenate((gt_frame, recon_frame, sample_frame), axis=0)
        append_frame_to_vid(combined_frame, vid_out)
        
    vid_out.release()


if __name__ == '__main__':
    model = "SIREN"
    models = ['SIREN']  # ['mlp', 'siren', 'ffn']
    data_idx = '07'
    data_idxs = ["05", "07", "14"] # ['07', '14', '15', '17'] # , '05', '18', '20', '21
    optimizer = "adam"
    scheduler = 'step'
    lr_scheduler = 'cosine'
    strategy = 'incremental'
    topk = '1'
    mt_ratio = ['0.2']
    # for dense_constant_constant
    # wandb_dict = {"05": "run-20240526_125656-ly5ny79v", "07": "run-20240526_125802-o706ay8y", "17": "run-20240526_125824-v288uofp"}
    # for incremental_step_constant
    wandb_dict = {"05": "run-20240526_235656", "07": "run-20240526_235648", "14": "run-20240526_235640"}
    # for incremental_step_cosine

    for model in models:
        for data_idx in data_idxs:
            for ratio in mt_ratio:
                data_path = f"../datasets/kodak/kodim{data_idx}.png"
                model_name = f"{optimizer}_{model}_mt{ratio}_{strategy}_{scheduler}_{lr_scheduler}_topk{topk}_kodim{data_idx}"
                sample_history_path = "logs/sampling/%s.pkl" % model_name
                loss_history_path = "logs/loss/%s.pkl" % model_name
                data_path = "../datasets/kodak/kodim%s.png" % data_idx
                superimpose_vid_path = "vis/dynamics/%s_superimpose.mp4" % model_name
                superimpose_vid_dir = "vis/dynamics/%s_superimpose" % model_name
                recon_vid_path = "vis/dynamics/%s_recon.mp4" % model_name
                recon_vid_dir = "vis/dynamics/%s_recon" % model_name
                sampling_vid_path = "vis/dynamics/%s_sampling.mp4" % model_name
                sampling_vid_dir = "vis/dynamics/%s_sampling" % model_name
                combined_vid_path = "vis/dynamics/%s_combined.mp4" % model_name
                combined_vid_dir = "vis/dynamics/%s_combined" % model_name
                
                loss_vid_path = "vis/dynamics/loss/%s_loss_dynamics.mp4" % model_name
                iou_curve_path = "vis/iou/%s_iou_curve.png" % model_name
                if data_idx == '17':
                    H, W = 768, 512
                else:
                    H, W = 512, 768
                data_range = 1

                with open(sample_history_path, 'rb') as f:
                    sample_history = pickle.load(f)
                with open(loss_history_path, 'rb') as f:
                    loss_history = pickle.load(f)

                print("Converting superimposed history to video...")
                mt_vid(sample_history, data_path, superimpose_vid_path, superimpose_vid_dir, H, W, fps=2)
                print("Superimposed history video completed.")

                print("Converting recon images to video...")
                wandb_dir_name = glob.glob(f"wandb/{wandb_dict[data_idx]}*")[0]
                wandb_log_path = f"{wandb_dir_name}/files/media/images"
                print(wandb_log_path)
                recon_vid(wandb_log_path, recon_vid_path, recon_vid_dir, H, W, fps=2)
                print("Recon images video completed.")

                print("Converting sampling history to video...")
                sample_vid(sample_history, sampling_vid_path, sampling_vid_dir, H, W, fps=2)
                print("Sampling history video completed.")

                print("Combining GT, sample history, and reconstruction history to video...")
                combined_vid(data_path, wandb_log_path, sample_history, combined_vid_path, combined_vid_dir, H, W, fps=2)
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
