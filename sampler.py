import math
import torch
import pickle


def mt_sampler(data, y, preds, size):
    # Given size is ratio of training data
    if math.isclose(size, 1.0):
        return data, y, None
    elif type(size) == float:
        n = int(size * len(data))
    # Given size is actual number of training data
    else:
        n = int(size)

    # mt sampling (returns indices)
    dif = torch.sum(torch.abs(y-preds), 1)
    _, idx = torch.topk(dif, n)

    # get sampled data
    sampled_data = data[idx]
    sampled_y = y[idx]
    
    return sampled_data, sampled_y, idx


def save_samples(sample_history, step, max_steps, samples, file_name):
    sample_history[str(step)] = samples.detach().cpu().numpy()
    with open(file_name, 'wb') as f:
        pickle.dump(sample_history, f)