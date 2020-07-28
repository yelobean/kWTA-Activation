import torchvision
import torchvision.transforms as transforms
import sys
from free_lunch.attack import *
sys.path.append('..')
from utils import *

def dataloader(batch_size=128, is_preprocessing=True):
    # Data loadder
    print('==> Preparing data..')
    if is_preprocessing:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/home/yelobean/dataset', train=True, download=False,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='/home/yelobean/dataset', train=False, download=False,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    return trainloader, testloader


# train function



def model_disc_train(net, net_low, net_high, ori_net, trainloader, device, optimizer, criterion, attack='', make_inputs_with_same_data=True):
    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        batchsize = inputs.shape[0]

        # make two augmented inputs for making shuffled concat inputs
        inputs_aug = inputs
        if not make_inputs_with_same_data:
            inputs_aug = random_flip(inputs)
            inputs_aug = random_crop(inputs_aug, 4)
            inputs = random_flip(inputs)
            inputs = random_crop(inputs, 4)

        # get layer output
        low_layer = net_low(inputs)
        high_layer = net_high(inputs_aug)

        # low_layer = torch.cat((low_layer, low_layer), 1)
        # low_layer = torch.cat((low_layer, low_layer), 1)
        # low_layer = torch.cat((low_layer, low_layer), 1)

        # make random number
        random_num = torch.randint(1, batchsize, (batchsize,)).to(device)
        arange_num = torch.arange(0, batchsize).to(device)
        random_num = (random_num + arange_num) % batchsize

        # concat layer outputs
        inputs_joint = concat_tensor_and_vector(low_layer, high_layer, device)
        inputs_marginal = concat_tensor_and_vector(low_layer, high_layer[random_num], device)

        # concat new inputs
        input_concat = torch.cat((inputs_joint, inputs_marginal), 0)

        # make new targets
        targets = make_target(batchsize, device, half_label=True)

        optimizer.zero_grad()
        outputs = net(input_concat)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

def model_adv_disc_train(net, net_low, net_high, ori_net, trainloader, device, optimizer, criterion, attack='', make_inputs_with_same_data=True):
    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batchsize = inputs.shape[0]

        # make two augmented inputs for making shuffled concat inputs
        if attack == 'fgsm':
            adv_data = fgsm_attack_rev(ori_net, inputs, targets)
        elif attack == 'bim':
            adv_data = basic_iterative_attack(ori_net, inputs, targets)
        elif attack == 'cw':
            adv_data = cw_l2_attack(ori_net, inputs, targets)
        elif attack == 'pgd':
            adv_data = pgd_attack(ori_net, inputs, targets, iters=10)
        else:
            adv_data = inputs
            raise print('ERROR!: No ADV!!')

        inputs_aug = inputs
        if not make_inputs_with_same_data:
            inputs_aug = random_flip(inputs)
            inputs_aug = random_crop(inputs_aug, 4)
            inputs = random_flip(inputs)
            inputs = random_crop(inputs, 4)

        # get layer output
        low_layer = net_low(inputs)
        high_layer = net_high(inputs_aug)

        # low_layer = torch.cat((low_layer, low_layer), 1)
        # low_layer = torch.cat((low_layer, low_layer), 1)
        # low_layer = torch.cat((low_layer, low_layer), 1)

        # get adv layer output
        adv_low_layer = net_low(adv_data)
        adv_high_layer = net_high(adv_data)

        # adv_low_layer = torch.cat((adv_low_layer, adv_low_layer), 1)
        # adv_low_layer = torch.cat((adv_low_layer, adv_low_layer), 1)
        # adv_low_layer = torch.cat((adv_low_layer, adv_low_layer), 1)

        # concat layer outputs
        inputs_joint = concat_tensor_and_vector(low_layer, high_layer, device)
        inputs_marginal = concat_tensor_and_vector(adv_low_layer, adv_high_layer, device)

        # concat new inputs
        input_concat = torch.cat((inputs_joint, inputs_marginal), 0)

        # make new targets
        targets = make_target(batchsize, device, half_label=True)

        optimizer.zero_grad()
        outputs = net(input_concat)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

def model_disc_test(net, net_low, net_high, ori_net, testloader, device, attack=''):
    global best_acc
    net.eval()
    ori_net.eval()
    attacked_correct = 0
    defenced_correct = 0
    total = 0


    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batchsize = inputs.shape[0]

        if attack == 'fgsm':
            adv_data = fgsm_attack_rev(ori_net, inputs, targets)
        elif attack == 'bim':
            adv_data = basic_iterative_attack(ori_net, inputs, targets)
        elif attack == 'cw_l2':
            adv_data = cw_l2_attack(ori_net, inputs, targets)
        elif attack == 'cw_linf':
            adv_data = cw_linf_attack(ori_net, inputs, targets)
        elif attack == 'pgd':
            adv_data = pgd_attack(ori_net, inputs, targets)
        elif attack == 'deepfool':
            adv_data = deepfool(ori_net, inputs, targets)
        elif attack == '':
            adv_data = inputs
        else:
            raise print('attack name error')

        with torch.no_grad():
            attacked_outputs = ori_net(adv_data)

        adv_data = adv_data.detach()
        targets_joint = make_target(batchsize, device, half_label=False)
        defenced_data = pgd_defense(net, net_low, net_high, adv_data, targets_joint, iters=20, random_start=True)
        # defenced_data = cw_l2_defense(net, net_low, net_high, adv_data, targets_joint)
        with torch.no_grad():
            defenced_outputs = ori_net(defenced_data)

        total += targets.size(0)

        _, predicted = attacked_outputs.max(1)
        attacked_correct += predicted.eq(targets).sum().item()

        _, predicted = defenced_outputs.max(1)
        defenced_correct += predicted.eq(targets).sum().item()


    print('Attacked Acc: %.3f%% (%d/%d)' % (100. * attacked_correct / total, attacked_correct, total))
    print('Defensed Acc: %.3f%% (%d/%d)' % (100. * defenced_correct / total, defenced_correct, total))
