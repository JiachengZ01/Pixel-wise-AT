from __future__ import print_function
import os
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchattacks import PGD, AutoAttack, CW

from model import ResNet18
# from models.wideresnet import WideResNet
# from models.preactresnet import PreActResNet18
# from models.vggnet import VGGNet19

# from utils.standard_loss import standard_loss
from torch.autograd import Variable
from cifar10 import CIFAR10
# from utils.data import data_dataset

from torchcam.methods import GradCAM, CAM
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=91, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--max-epsilon', default=12/255,
                    help='maximum allowed perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ResNet_18/PAT',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def craft_weight_matrix(model, data, device):
    colormap = 'jet'
    batch = data.shape[0]
    img_size1 = data.shape[-2]
    img_size2 = data.shape[-1]

    weight_matrix_tensor = torch.empty(batch, 3, img_size1, img_size2).to(device)

    cam_extractor = GradCAM(model.module, 'layer4')

    for i in range(batch):
        output = model(data[i].unsqueeze(0))

        heatmap = cam_extractor(output.argmax().item(), output)
        mask = to_pil_image(heatmap[0].squeeze(0).cpu().numpy())
        cmap = cm.get_cmap(colormap)

        # Resize mask and apply colormap
        overlay = mask.resize((img_size1, img_size2), resample=Image.BICUBIC)
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.double)

        normalized_overlay = overlay / 255

        # Scale the normalized values to have a mean of 0 and a standard deviation of 1
        mean = np.mean(normalized_overlay)
        std = np.std(normalized_overlay)
        weight_matrix = torch.from_numpy((normalized_overlay - mean) / std)
        weight_matrix = torch.clamp(weight_matrix, 1, weight_matrix.max()).float()
        weight_matrix = weight_matrix.permute(2, 0, 1).to(device)
        weight_matrix_tensor[i] = weight_matrix

    # clear the hooks on model
    cam_extractor.remove_hooks()

    return weight_matrix_tensor

def generate_weighted_eps(weight_matrix):
    epsilon = torch.where(weight_matrix > 1, args.max_epsilon, args.epsilon)
    return epsilon

def train(args, model, device, train_loader, optimizer, epoch):

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        X, y = Variable(data, requires_grad=True), Variable(label)

        # calculate robust loss
        model.eval()

        weight_matrix = craft_weight_matrix(model, data, device)

        weighted_eps = generate_weighted_eps(weight_matrix)

        data = _pgd_whitebox(model,
                             X,
                             y,
                             weighted_eps,
                             epsilon=args.epsilon,
                             num_steps=args.num_steps,
                             step_size=args.step_size)

        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = F.cross_entropy(out, label)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))

def craft_adversarial_example(model, x_natural, y,
                              step_size=2/255, epsilon=8/255, perturb_steps=10,
                              mode='pgd'):

    if mode == 'pgd':
        attack = PGD(model, eps=epsilon, alpha=step_size, steps=perturb_steps, random_start=True)
        x_adv = attack(x_natural, y)
    elif mode == 'cw':
        attack = CW(model, c=1, kappa=0, steps=10, lr=0.01)
        x_adv = attack(x_natural, y)
    elif mode == 'aa':
        attack  = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
        x_adv = attack(x_natural, y)

    '''
    adversary = DDNL2Attack(model, nb_iter=40, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                            clip_max=1.0, targeted=False, loss_fn=None)
    x_adv = attack(x_natural, y)
    '''

    '''
    adversary = LinfFWA(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                      eps=8/255, kernel_size=4, lr=0.007, nb_iter=40, dual_max_iter=15, grad_tol=1e-4,
                    int_tol=1e-4, device="cuda", postprocess=False, verbose=True)
    x_adv = adversary(x_natural, y)
    # x_adv = adversary.perturb(x_natural, y)
    '''

    '''
    adversary = SpatialTransformAttack(model, 10, clip_min=0.0, clip_max=1.0, max_iterations=10, search_steps=5, targeted=False)
    x_adv = adversary(x_natural, y)
    '''

    '''
    adversary = TIDIM_Attack(model,
                       decay_factor=1, prob=0.5,
                       epsilon=8/255, steps=40, step_size=0.01,
                       image_resize=33,
                       random_start=False)
    '''

    '''
    adversary = TIDIM_Attack(eps=8/255, steps=40, step_size=0.007, momentum=0.1, prob=0.5, clip_min=0.0, clip_max=1.0,
                 device=torch.device('cuda'), low=32, high=32)
    x_adv = adversary.perturb(model, x_natural, y)
    '''

    return x_adv


def eval_test(model, device, test_loader, attack='pgd'):
    model.eval()
    correct = 0
    correct_adv = 0

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        X, y = Variable(data, requires_grad=True), Variable(label)

        logits_out = model(data)
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

        data = craft_adversarial_example(model=model, x_natural=X, y=y,
                                         step_size=args.step_size, epsilon=args.epsilon, perturb_steps=20,
                                         mode=attack)

        logits_out = model(data)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv += pred.eq(label.view_as(pred)).sum().item()

    print('Test: Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), correct_adv,
        len(test_loader.dataset), 100. * correct_adv / len(test_loader.dataset)))


def clamp(eta, epsilon):
    # Element-wise clamp using the epsilon tensor
    eta_clamped = torch.where(eta > epsilon, epsilon, eta)
    eta_clamped = torch.where(eta < -epsilon, -epsilon, eta_clamped)
    return eta_clamped


def _pgd_whitebox(model,
                  X,
                  y,
                  weighted_eps,
                  epsilon=8/255,
                  num_steps=10,
                  step_size=2/255):
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = clamp(X_pgd.data - X.data, weighted_eps)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def main():
    # settings
    setup_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # setup data loader
    train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()
    test_loader = CIFAR10(test_batch_size=args.batch_size).test_data()

    # init model, ResNet18() can be also used here for training
    model = ResNet18().to(device)
    # model = PreActResNet18(10).to(device)
    # model = WideResNet(34, 10, 10).to(device)
    # model = VGGNet19().to(device)

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'best_model.pth'))
            print('save the model')

        print('================================================================')

    #model.load_state_dict(torch.load('checkpoint/ResNet_18/PAT/pat.pth'))
    # evaluation on natural examples
    print('================================================================')

    eval_test(model, device, test_loader, attack='pgd')
    print('================================================================')
    eval_test(model, device, test_loader, attack='cw')
    print('================================================================')
    eval_test(model, device, test_loader, attack='aa')


if __name__ == '__main__':
    main()
