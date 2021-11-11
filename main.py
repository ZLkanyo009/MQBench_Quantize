'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization

from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse

from models import *
from utils import progress_bar, choose_model, choose_backend


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0

    if args.quantize:
        net.eval()
        enable_calibration(net)
        calibration_flag = True

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        if args.quantize:
            if calibration_flag:
                if batch_idx >= 0:
                    calibration_flag = False
                    net.zero_grad()
                    net.train()
                    enable_quantization(net)
                else:
                    continue

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    print("============== eval ==================")
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    #if True:
    if acc > best_acc:
        if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        else:
            print("no")
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
            
        if args.quantize:
            torch.save(state, './checkpoint/' + args.model + '_ckpt_q.pth')
            print('Saved in ./checkpoint/' + args.model + '_ckpt_q.pth')
        else:
            torch.save(state, './checkpoint/' + args.model + '_ckpt.pth')
            print('Saved in ./checkpoint/' + args.model + '_ckpt.pth')
        best_acc = acc

parser = argparse.ArgumentParser(description='PyTorch MQBench Quantization Aware Training')
parser.add_argument('model', type=str,
                    help='network model type: see detail in folder ''models'' ')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--quantize", '-q', action='store_true',
                    help='fp32 train or low_bit QAT, -q means low_bit QAT')
parser.add_argument("--parallel", '-p', default = None ,type=str,
                    help='choose DP or DDP')
parser.add_argument("--BackendType", '-BT', default = 'Tensorrt' ,type=str,
                    help='choose device to deploy')
parser.add_argument("--local_rank", type=int,
                    help='When there is a host slave situation in DDP,\
                          the host is local_ rank = 0')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = choose_model(args)
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ args.model +'_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    if not args.quantize:
        best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.quantize:
    net.train()
    backend = choose_backend(args)
    net = prepare_qat_fx_by_platform(net, backend)
#assert False
net = net.to(device)

if device == 'cuda' and torch.cuda.device_count() > 1 and args.parallel == 'DP':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if device == 'cuda' and torch.cuda.device_count() > 1 and args.parallel == 'DDP':
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
        net = DDP(net, find_unused_parameters=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
