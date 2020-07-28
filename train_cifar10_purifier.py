### if want insert noise, change model_train to test_model_train
### control second net(:2, :3, :4) & load file name
import argparse
import time
import sys
import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet18

from free_lunch.functions import *
from free_lunch.purifier_resnet import Discriminator18, Discriminator34
# from free_lunch.purifier_resnet import Discriminator18, Discriminator34
from kWTA import resnet
from kWTA import wideresnet

sys.path.append('..')
from utils import *

# set parser args
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
parser.add_argument('--max_epoch', '-m_e', type=int, default=10, help='max epoch')
parser.add_argument('--test_interval', type=int, default=5, help='max epoch')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, help='the parameter of l2 restriction for weights')

# train net spec
parser.add_argument('--attack', '-a', default='', help='which attack? option:(blank, fgsm, bim, cw, pgd)')
parser.add_argument('--make_inputs_with_same_data', '-msd', type=bool, default='T', help='Are inputs of low/ high net different?')

# etc option
parser.add_argument('--gpu', '-g', default='9', help='which gpu to use')
parser.add_argument('--is_test', '-test', type=bool, default='T', help='only test')

# load net spec
parser.add_argument('--which_AT', '-at', default='trades', help='which AT (nat, at, trades)')
parser.add_argument('--is_kWTA', '-wta', type=bool, default='', help='option WTA on/off')
parser.add_argument('--is_Wide', '-wide', type=bool, default='', help='option WIDE on/off')
parser.add_argument('--iters', '-i', default=10, type=int, help='how many attack iters')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# set visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# torch cuda available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader, testloader = dataloader(batch_size=args.batch_size // 2, is_preprocessing=args.make_inputs_with_same_data)



name = ''
if args.is_Wide:
    name = name + '_wide'
if args.is_kWTA:
    name = name + '_kWTA'
name = name + '_' + args.which_AT
name = name + '_iter' + str(args.iters)

if args.is_Wide:
    if args.is_kWTA:
        load_net = wideresnet.SparseWideResNet(depth=34, num_classes=10, widen_factor=10, sp=0.1, sp_func='vol').to(device)
    else:
        load_net = wideresnet.WideResNet(depth=34, num_classes=10, widen_factor=10).to(device)
else:
    if args.is_kWTA:
        load_net = resnet.SparseResNet18(sparsities=[0.1, 0.1, 0.1, 0.1], sparse_func='vol').to(device)
    else:
        load_net = resnet.ResNet18().to(device)

if len(args.gpu) > 2:
    import torch.backends.cudnn as cudnn
    if device == 'cuda':
        load_net = torch.nn.DataParallel(load_net)
        cudnn.benchmark = True

load_net.load_state_dict(torch.load('models/resnet18_cifar' + name + '.pth', map_location=device))

if len(args.gpu) > 2:
    load_net = list(load_net.children())[0]

# make net don't requires grad
for p in load_net.parameters():
    p.requires_grad = False

# print(load_net)
if args.is_Wide:
    net_temp = list(load_net.children())[1]
    for i in range(2):
        net_temp = list(net_temp.children())[0]
    net_low = torch.nn.Sequential(list(load_net.children())[0], *list(net_temp.children())[0:2]).to(device)
else:
    net_low = torch.nn.Sequential(*list(load_net.children())[0:3]).to(device) #0:1 0:2 0:3
net_high = torch.nn.Sequential(*list(load_net.children())[:-1], nn.AdaptiveAvgPool2d((1, 1))).to(device)


net_low.eval()
net_high.eval()
load_net.eval()

if args.is_Wide:
    num_input_layer_filter = 16 + 640
else:
    num_input_layer_filter = 64 + 512
net = Discriminator18(False, num_input_layer_filter=num_input_layer_filter).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1) #default: 100, 150/ full 200

for epoch in range(0, args.max_epoch):
    print('\nEpoch: %d' % epoch)
    print('gpu:', args.gpu)
    print('load_name:', name)
    print('same_input:', args.make_inputs_with_same_data)
    print('used attack:', args.attack)
    print('is TEST?:', args.is_test)
    print('preX')

    start_time = time.time()
    if not args.is_test:
        if args.attack == '':
            model_disc_train(net, net_low, net_high, load_net, trainloader, device, optimizer, criterion, attack='', make_inputs_with_same_data=args.make_inputs_with_same_data)
        else:
            model_adv_disc_train(net, net_low, net_high, load_net, trainloader, device, optimizer, criterion, attack=args.attack, make_inputs_with_same_data=args.make_inputs_with_same_data)
    # else:
    #     net.load_state_dict(torch.load("purifier_models/loadfrom" + name + "_sameinput" + str(args.make_inputs_with_same_data) + "_advtrain" + args.attack + ".pth", map_location=device))

    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Tr Time:', times)

    start_time = time.time()
    # if epoch % args.test_interval == 0:
    # if not args.is_test:
    print('clean ACC')
    model_disc_test(net, net_low, net_high, load_net, testloader, device, attack='')
    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Clean Te Time:', times)

    start_time = time.time()
    print('FGSM ACC')
    model_disc_test(net, net_low, net_high, load_net, testloader, device, attack='fgsm')
    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Attack Te Time:', times)

    start_time = time.time()
    print('pgd ACC')
    model_disc_test(net, net_low, net_high, load_net, testloader, device, attack='pgd')
    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Attack Te Time:', times)

    # if args.is_test:
    start_time = time.time()
    print('CW L2 ACC')
    model_disc_test(net, net_low, net_high, load_net, testloader, device, attack='cw_l2')
    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Attack Te Time:', times)

    start_time = time.time()
    print('CW LINF ACC')
    model_disc_test(net, net_low, net_high, load_net, testloader, device, attack='cw_linf')
    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Attack Te Time:', times)



    # start_time = time.time()
    # print('deepfool ACC')
    # model_disc_test(net, net_low, net_high, load_net, testloader, device, attack='deepfool')
    # times = str(datetime.timedelta(seconds=time.time() - start_time))
    # print('Attack Te Time:', times)

    scheduler.step(epoch)

    if args.is_test:
        break

    torch.save(net.state_dict(), "purifier_models/loadfrom" + name + "_sameinput" + str(args.make_inputs_with_same_data) + "_advtrain" + args.attack + ".pth")


