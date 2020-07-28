import os

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def onehot(targets_list, n):
    return torch.eye(n)[targets_list]

def makedirectory(str_dir):
    if not os.path.exists(str_dir):
        os.makedirs(str_dir)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)
    if isinstance(m, nn.Conv2d):
        # nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)

# def fgsm_attack(image, epsilon, data_grad, max_value, min_value):
#     sign_data_grad = data_grad.sign()
#     perturbed_image = image + epsilon*(max_value - min_value)*sign_data_grad
#     perturbed_image = torch.clamp(perturbed_image, min_value, max_value)
#
#     return perturbed_image

def fgsm_defence(image, epsilon, data_grad, max_value, min_value):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon*(max_value - min_value)*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, min_value, max_value)

    return perturbed_image

def inject_noise(image, epsilon, max_value, min_value):
    sign_data_grad = (torch.randint(0, 2, image.shape) - 0.5).sign().to(device='cuda')
    perturbed_image = image - epsilon*(max_value - min_value)*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, min_value, max_value)

    return perturbed_image

def concat_tensor_and_vector(tensor, vector, device):
    reshaped_vector = vector.reshape((vector.shape[0], vector.shape[1], 1, 1)) * torch.ones((tensor.shape[0], vector.shape[1], tensor.shape[2], tensor.shape[3])).to(device)
    concat_output = torch.cat((tensor, reshaped_vector), 1)

    return concat_output

def concat_tensor_and_int(tensor, integer, num_of_label, device):
    vector = onehot(integer, num_of_label).to(device)

    return concat_tensor_and_vector(tensor, vector, device)

def make_target(batchsize, device, half_label=True):
    if half_label:
        targets = torch.ones((batchsize * 2, 1)).to(device)
        targets[batchsize:] = targets[batchsize:] * 0
    else:
        targets = torch.ones((batchsize, 1)).to(device)

    return targets

def random_crop(tensor, padding):
    tensor_pad = F.pad(tensor, (padding, padding, padding, padding))

    for i in range(tensor.shape[0]):
        a, b = np.random.randint(0, padding * 2 + 1, 2)
        tensor[i] = tensor_pad[i, :, a:a + 32, b:b + 32]

    return tensor

def random_flip(tensor):
    a = np.random.randint(0, 2, 1)
    if a == 1:
        tensor = tensor.flip(dims=(3, ))

    return tensor