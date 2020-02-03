import argparse
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

import mnist_dataset
import models
import cifar_supervised_repr

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ntrain', type=int, default=60000)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()
cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


def kl_to_prior(means, sigmas, prior_means, prior_sigmas):
    means = torch.cat([m.flatten() for m in means])
    prior_means = torch.cat([m.flatten() for m in prior_means])

    sigmas = torch.cat([s.flatten() for s in sigmas])
    prior_sigmas = torch.cat([s.flatten() for s in prior_sigmas])

    mu_diff = means - prior_means
    mu_term = (mu_diff**2 / prior_sigmas**2).sum()

    sigma_tr_term = (sigmas**2 / (prior_sigmas**2)).sum()

    # take the log first to make everything linear,
    # then do the sum-of-differences instead of the difference-of-sums
    # to make the numerics better
    log_det_term = 2 * (torch.log(prior_sigmas) - torch.log(sigmas)).sum()

    k = means.shape[0]

    # group calculation of (sigma_tr + log_det - k) to avoid numerical issues;
    # each term is large but the difference is ~0
    kl = 0.5 * (mu_term + (sigma_tr_term + log_det_term - k))

    return kl, 0.5 * mu_term, 0.5 * sigma_tr_term, 0.5 * log_det_term, -0.5 * k


def differentiable_bound(empirical_risk,
                         means, sigmas, prior_means, prior_sigmas,
                         delta=(1-0.95), n=60000):

    kl, _, _, _, _ = kl_to_prior(means, sigmas, prior_means, prior_sigmas)
    kl_ratio = (kl + np.log(2 * np.sqrt(n) / delta)) / (2 * n)
    return ((empirical_risk + kl_ratio)**0.5 + (kl_ratio)**0.5)**2


def dziugaite_bound(empirical_risk, means, sigmas, prior_means, prior_sigmas,
                    delta=(1-0.95), n=60000):
    kl, _, _, _, _ = kl_to_prior(means, sigmas, prior_means, prior_sigmas)
    gap = ((kl + math.log(n / delta)) / (2 * (n - 1))) ** 0.5
    return empirical_risk + gap, kl


def KLdiv(pbar, p):
    return pbar * np.log(pbar/p) + (1-pbar) * np.log((1-pbar)/(1-p))


def KLdiv_prime(pbar, p):
    return (1-pbar)/(1-p) - pbar/p


def Newt(p, q, c):
    newp = p - (KLdiv(q, p) - c)/KLdiv_prime(q, p)
    return newp


def approximate_BPAC_bound(train_accur, B_init, niter=5):
    """Input B_init should be (KL + log_term) / n"""
    B_RE = 2 * B_init ** 2
    A = 1-train_accur
    B_next = B_init+A
    if B_next > 1.0:
        return 1.0
    for i in range(niter):
        B_next = Newt(B_next, A, B_RE)
    return B_next


def train_noisy(model, train_loader, optimizer, epoch, prior_means, prior_sigmas,
                objective='pbb', penalty=1):
    model.train()

    def compute_bound(pred_err, means, sigmas):
        if objective == 'dziugaite':
            bound, kl = dziugaite_bound(
                pred_err, means, sigmas, prior_means, prior_sigmas)
            return bound
        elif objective == 'kl':
            kl, _, _, _, _ = kl_to_prior(
                means, sigmas, prior_means, prior_sigmas)
            return pred_err + penalty * kl
        elif objective == 'pbb':
            return differentiable_bound(pred_err, means, sigmas, prior_means, prior_sigmas)
        else:
            assert False

    mean_pred_err = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # get data
        data, target = data.to(DEVICE), target.to(DEVICE)

        # calculate loss
        output = model(data)
        pred_err = F.nll_loss(output, target)
        mean_pred_err += pred_err.item()

        means = model.get_means()
        sigmas = model.get_sigmas()

        bound = compute_bound(pred_err, means, sigmas)

        # take step
        optimizer.zero_grad()
        bound.backward()
        optimizer.step()

    batch_idx += 1
    mean_pred_err /= batch_idx
    means = model.get_means()
    sigmas = model.get_sigmas()
    avg_surr_bound = compute_bound(mean_pred_err, means, sigmas)
    return 'Train Epoch: {} Batch: {} \t LR: {:.3e} \t Log loss: {:.6f}\t Surrogate bound: {:.6f}'.format(
        epoch, batch_idx, optimizer.param_groups[0]['lr'], mean_pred_err, avg_surr_bound)

def train_lagrangian(model, lambda_param, train_loader, optimizer, lambda_optimizer, epoch, prior_means, prior_sigmas, max_error=0.1):
    model.train()

    def main_loss(log_loss, means, sigmas):
        kl, _, _, _, _ = kl_to_prior(means, sigmas, prior_means, prior_sigmas)
        return lambda_param() * log_loss + kl / (len(train_loader) * args.batch_size)

    def lagrangian_loss(pred_err):
        return max(min(max_error - pred_err, 0.2), -0.2)

    samples = 0
    mean_log_loss = 0
    mean_lambda_loss = 0
    correct = 0

    # optimize the main loss
    for batch_idx, (data, target) in enumerate(train_loader):
        # get data
        data, target = data.to(DEVICE), target.to(DEVICE)

        # calculate loss
        output = model(data)
        log_loss = F.nll_loss(output, target)

        # calculate accuracy
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += batch_correct

        means = model.get_means()
        sigmas = model.get_sigmas()

        loss = main_loss(log_loss, means, sigmas)

        # take step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # take step on lambda
        lambda_optimizer.zero_grad()
        lambda_loss = lagrangian_loss(1 - (batch_correct / len(data)))
        lambda_pregrad = lambda_loss * lambda_param()
        lambda_pregrad.backward()
        lambda_optimizer.step()

        mean_log_loss += log_loss.item()
        mean_lambda_loss += lambda_loss
        samples += len(data)
    accuracy = correct / samples
    batch_idx += 1

    # logging
    mean_log_loss /= batch_idx
    mean_lambda_loss /= batch_idx
    means = model.get_means()
    sigmas = model.get_sigmas()
    return 'Train Epoch: {} Batch: {} \t LR: {:.3e} \t Log loss: {:.6f}\t Lambda: {:.6f}\t Lambda loss: {:.6f}'.format(
        epoch, batch_idx, optimizer.param_groups[0]['lr'], mean_log_loss, lambda_param().item(), mean_lambda_loss)


def eval_noisy(model, eval_loader, prior_means, prior_sigmas, name, length):
    model.eval()
    test_loss = torch.zeros(1).to(DEVICE)
    correct = 0
    samples = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum')
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            samples += len(data)

            if samples >= length:
                break

    test_loss /= samples
    incorrect = samples - correct

    means = model.get_means()
    sigmas = model.get_sigmas()

    test_risk = incorrect / samples
#     ce_bound = differentiable_bound(test_loss, means, sigmas, prior_means, prior_sigmas)
    dz_bound, _ = dziugaite_bound(
        test_risk, means, sigmas, prior_means, prior_sigmas, n=length)
    pbb_bound = differentiable_bound(
        test_risk, means, sigmas, prior_means, prior_sigmas, n=length)
    kl, mean_term, sigma_tr_term, log_det_term, k_term = kl_to_prior(
        means, sigmas, prior_means, prior_sigmas)

    flat_sigmas = torch.cat([p.flatten() for p in model.get_sigmas()], 0)
    flat_prior_sigmas = torch.cat([p.flatten() for p in prior_sigmas], 0)
    return ('{} set: Error: {}/{} ({:.0f}%), Log loss: {:.6f}, Dziugaite bound: {:.6f}, PBB bound: {:.6f}, '
            'KL: {:.6f}, Mean term: {:.6f}, Sigma_tr term: {:.6f}, Log-det term: {:.6f}, k term: {:.6f}').format(
        name,
        incorrect, samples, 100. * incorrect / \
        samples, test_loss.item(), dz_bound.item(), pbb_bound.item(),
        kl, mean_term, sigma_tr_term, log_det_term, k_term)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    mnist_dataset.MNIST('../data', train=True, download=True, n_examples=args.ntrain,
                        transform=transforms.Compose([
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    mnist_dataset.MNIST('../data', train=False, download=True, n_examples=10000,
                        transform=transforms.Compose([
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


prior_sigma = 3e-2

# lambda_param = models.Lagrangian(1e-3).to(DEVICE)
# lambda_optimizer = optim.SGD(lambda_param.parameters(), lr=1e-2)
model = models.NoisyNet(prior_sigma, per_sample=True,
                        clipping='hard').to(DEVICE)

prior_means = [p.clone().detach() for p in model.get_means()]
prior_sigmas = [p.clone().detach() for p in model.get_sigmas()]

optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.95)
# optimizer = optim.Adadelta(model.parameters(), lr=1e-1)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.3)

for epoch in range(0, 10000):
    train_str = train_noisy(model, train_loader, optimizer,
                            epoch, prior_means, prior_sigmas, objective='pbb')
    if epoch % 1 == 0:
        train_eval_str = eval_noisy(
            model, train_loader, prior_means, prior_sigmas, "Train", args.ntrain)
        test_eval_str = eval_noisy(
            model, test_loader, prior_means, prior_sigmas, "Test", 10000)
        print("Epoch:", epoch)
        print(train_str)
        print(train_eval_str)
        print(test_eval_str)
        print()

