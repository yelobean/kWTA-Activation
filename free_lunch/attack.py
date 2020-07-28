from __future__ import print_function
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')
from utils import *

# epsilons = [0, .05, .1, .15, .2, .25, .3]
# use_cuda=True

# you need to set requires_grad attribute of tensor before calling this function.
# data.requires_grad = True
# you can call this function as #perturbed_data = fgsm_attack(data, epsilon, data_grad)
# FGSM attack code
def cw_l2_defense(model, model_low, model_high, images, labels, targeted=False, c=1e-4, kappa=50, max_iter=40, learning_rate=0.01, device='cuda'):

    # Define f-function
    def f(x):
        # outputs = model(x)
        low_layer = model_low(images)
        high_layer = model_high(images)

        # low_layer = torch.cat((low_layer, low_layer), 1)
        # low_layer = torch.cat((low_layer, low_layer), 1)
        # low_layer = torch.cat((low_layer, low_layer), 1)

        inputs_joint = concat_tensor_and_vector(low_layer, high_layer, device)
        outputs = model(inputs_joint)

        return outputs

        # one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
        #
        # i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        # j = torch.masked_select(outputs, one_hot_labels.bool())
        #
        # # If targeted, optimize for making the other class most likely
        # if targeted:
        #     return torch.clamp(i-j, min=-kappa)
        #
        # # If untargeted, optimize for making the other class most likely
        # else:
        #     return torch.clamp(j-i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)
    w.detach_()
    w.requires_grad = True

    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10

    for step in range(max_iter):

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return (1/2*(nn.Tanh()(w) + 1)).detach()
            prev = cost

        # print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1).detach()

    return attack_images

def pgd_defense(model, model_low, model_high, images, labels, targeted=False, eps=8/255, alpha=2/255, iters=20, random_start=True, device='cuda', detector_on=True):

    loss = nn.BCELoss()
    if targeted:
        loss = lambda x, y: -nn.BCELoss()(x, y)

    ori_images = images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        images = images + torch.empty_like(images).uniform_(-eps, eps)
        images = torch.clamp(images, min=0, max=1)

    for i in range(iters):
        images.requires_grad = True

        low_layer = model_low(images)
        high_layer = model_high(images)

        # low_layer = torch.cat((low_layer, low_layer), 1)
        # low_layer = torch.cat((low_layer, low_layer), 1)
        # low_layer = torch.cat((low_layer, low_layer), 1)

        inputs_joint = concat_tensor_and_vector(low_layer, high_layer, device)
        outputs = model(inputs_joint)

        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        if detector_on: # and (i == 0):
            grad = grad * torch.where(outputs > 0.5, torch.tensor([1.]).to(device), torch.tensor([0.]).to(device)).reshape((-1, 1, 1, 1))

        adv_images = images - alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    adv_images = images

    return adv_images

def fgsm_attack_rev(model, images, labels, eps=8/255):
    loss = nn.CrossEntropyLoss()

    images.requires_grad = True

    outputs = model(images)

    cost = loss(outputs, labels)

    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images



# you can call this function as
# perturbed_data = basic_iterative_attack(model, loss, data, target, scale=1, eps=4, alpha=epsilon)
#BIM attack code
def basic_iterative_attack(model, images, labels, eps=8/255, alpha=2/255, iters=0):

    if iters == 0 :
        # The paper said min(eps + 4, 1.25*eps) is used as iterations
        iters = int(min(eps * 255 + 4, 1.25 * eps * 255))

    loss = nn.CrossEntropyLoss()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        # model.zero_grad()
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        # cost.backward()

        adv_images = images + alpha * grad.sign()

        # Clip attack images(X')
        # min{255, X+eps, max{0, X-eps, X'}}
        # = min{255, min{X+eps, max{max{0, X-eps}, X'}}}

        # a = max{0, X-eps}
        a = torch.clamp(images - eps, min=0)
        # b = max{a, X'}
        b = (adv_images>=a).float()*adv_images + (a>adv_images).float()*a
        # c = min{X+eps, b}
        c = (b > images+eps).float()*(images+eps) + (images+eps >= b).float()*b
        # d = min{255, c}
        images = torch.clamp(c, max=1).detach_()

    return images


def cw_linf_attack(model, images, labels, targeted=False, c=1e-2, kappa=0, max_iter=20, learning_rate=0.01, device='cuda'):

    # Define f-function
    def f(x):
        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i-j, min=-kappa)

        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j-i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)
    w.detach_()
    w.requires_grad = True

    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10

    for step in range(max_iter):
        loss1 = torch.sum(c*f(images + w))
        loss2 = torch.sum(torch.relu(w-0.031))

        # a = 1/2*(nn.Tanh()(w) + 1)
        #
        # loss1 = nn.MSELoss(reduction='sum')(a, images)
        # loss2 = torch.sum(c*f(a))
        #
        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0:
            if cost > prev:
                # print('Attack Stopped due to CONVERGENCE....')
                return images + torch.clamp(w, max=0.031)
            prev = cost

        # print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = images + torch.clamp(w, max=0.031)

    return attack_images


# CW-L2 Attack
# you can call this function as
# perturbated_images = cw_l2_attack(model, images, labels, targeted=False, c=0.1)
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
def cw_l2_attack(model, images, labels, targeted=False, c=1e-2, kappa=0, max_iter=30, learning_rate=0.01, device='cuda'):

    # Define f-function
    def f(x):
        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i-j, min=-kappa)

        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j-i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)
    w.detach_()
    w.requires_grad = True

    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10

    for step in range(max_iter):

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return (1/2*(nn.Tanh()(w) + 1)).detach()
            prev = cost

        # print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1).detach()

    return attack_images

def pgd_attack(model, images, labels, targeted=False, eps=8/255, alpha=1/255, iters=20, random_start=True):

    loss = nn.CrossEntropyLoss()
    if targeted:
        loss = lambda x, y: -nn.CrossEntropyLoss()(x, y)

    ori_images = images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        images = images + torch.empty_like(images).uniform_(-eps, eps)
        images = torch.clamp(images, min=0, max=1)

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    adv_images = images

    return adv_images

def deepfool(model, images, labels, steps=20):

    for b in range(images.shape[0]):

        image = images[b:b + 1, :, :, :]

        image.requires_grad = True
        output = model(image)[0]

        _, pre_0 = torch.max(output, 0)
        f_0 = output[pre_0]
        grad_f_0 = torch.autograd.grad(f_0, image,
                                       retain_graph=False,
                                       create_graph=False)[0]
        num_classes = len(output)

        for i in range(steps):
            image.requires_grad = True
            output = model(image)[0]
            _, pre = torch.max(output, 0)

            if pre != pre_0:
                image = torch.clamp(image, min=0, max=1).detach()
                break

            r = None
            min_value = None

            for k in range(num_classes):
                if k == pre_0:
                    continue

                f_k = output[k]
                grad_f_k = torch.autograd.grad(f_k, image,
                                               retain_graph=True,
                                               create_graph=True)[0]

                f_prime = f_k - f_0
                grad_f_prime = grad_f_k - grad_f_0
                value = torch.abs(f_prime) / torch.norm(grad_f_prime)

                if r is None:
                    r = (torch.abs(f_prime) / (torch.norm(grad_f_prime) ** 2)) * grad_f_prime
                    min_value = value
                else:
                    if min_value > value:
                        r = (torch.abs(f_prime) / (torch.norm(grad_f_prime) ** 2)) * grad_f_prime
                        min_value = value

            image = torch.clamp(image + r, min=0, max=1).detach()

        images[b:b + 1, :, :, :] = image

    adv_images = images

    return adv_images

def test( model, device, test_loader, epsilon, attack_method ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        print(init_pred)
        print(target)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        if attack_method == 'cw_l2_attack':
            perturbed_data = cw_l2_attack(device, model, data, target, targeted=False, c=0.3)
        elif  attack_method == 'fgsm':
            perturbed_data = fgsm_attack(data, epsilon, data_grad )
        elif attack_method =='bim':
            perturbed_data = basic_iterative_attack(model, loss, data, target, scale=1, eps=4, alpha=epsilon)


        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
