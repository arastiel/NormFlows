from torch.utils.data import Dataset, DataLoader
import os
import datetime
import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

from NICE import NICE
from RealNVP import RealNVP
from RealNVP2 import RealNVP2
from utilfuncs import load_checkpoint, save_checkpoint, sample, pre_process, prep_dataset, show_dist
import args

dataset = args.dataset
img_size = args.img_size
channel = args.channel
batch_size = args.batch_size
filename = args.filename


train_loader, test_loader = prep_dataset(dataset)

def loss_function(target_distribution, z, log_det_jacobian, log_pre):
    """

    exact log likelihood defined by change of variables
    logpX(x) = logpZ(z) + logdet_jac (+ logprepro)

    :param target_distribution: target_distr
    :param z: transformed x
    :param log_det_jacobian: jacobian of transformation
    :param log_pre: jacobian of prepro
    :return: average loss per image + statistics
    """
    #print(torch.mean(z))
    #z = torch.clamp(z, 0., 1.)
    log_pZ_z = target_distribution.log_prob(z).to(device)
    #log_pZ_z = log_pZ_z.view(log_pZ_z.size(0), -1).sum(-1) - np.log(256)*28*28
    #log_det_jacobian = log_det_jacobian.view(log_det_jacobian.size(0), -1).sum(-1)
    log_likelihood = log_pZ_z + log_det_jacobian + log_pre
    loss = -log_likelihood
    return torch.div(loss.sum(), batch_size), log_pZ_z.sum(), log_det_jacobian.sum()
    #return loss.mean(), log_pZ_z.mean(), log_det_jacobian.mean()
    # average sum loss per pic, sum likelihood z in Z, sum log det jacobian


    # print(log_det_jacobian.sum()/batch_size)
    # bpd = -log_likelihood.sum(dim=[1,2,3]).mean() * np.log2(np.exp(1)) / np.prod(z.shape[1:])
    # print(bpd)
    # return -log_likelihood.sum()/batch_size
    # return -log_likelihood.sum(dim=[1, 2, 3]).mean()  # .mean()

def train(model, train_loader, optimizer, target_distribution):
    """

    train function

    :param model: model to train
    :param train_loader: processed train data
    :param optimizer: used optimizer
    :param target_distribution: target distribution
    :return: -
    """
    size = len(train_loader)
    model.train()
    bpds_ = []
    for i, x in enumerate(train_loader):
        # x = x.to(device)
        x = x[0].to(device)
        x, log_pre = pre_process(x, False) #PREPROCESS OUT OF RealNVP.py
        z, log_dz_by_dx = model(x)
        loss, log_pZ_z, log_det_jac = loss_function(target_distribution, z, log_dz_by_dx, log_pre)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            #bpd = (loss.item() + np.log(256.) * (channel*img_size*img_size)) \
            #    / ((channel*img_size*img_size) * np.log(2.))
            bpd2 = (loss.item() / ((channel*img_size*img_size) * np.log(2.)))
            print('test {}'.format(loss-log_pZ_z/batch_size))
            #print('loss: {}, loss.item: {}'.format(loss, loss.item()))
            print('Average Loss at iteration {}/{} is {}'.format(i, size, loss.cpu().item()))
            print('log_det sum is {}'.format(log_det_jac))
            print('log_prob(z) sum is {}'.format(log_pZ_z))
            #loss +=  np.log(255)
            #loss += log_pre.sum()/batch_size
            #print('BPD at iteration {}/{} is {}'.format(i, size,
            #                                           #loss.cpu().item() /
            #                                            loss/
            #                                           (channel*img_size*img_size) /
            #                                           np.log(2)))
            #print('test {}'.format((loss+np.log(256.)) * (channel*img_size*img_size) / ((channel*img_size*img_size) * np.log(2))))
            #print('BPD at iteration {}/{} is {}'.format(i, size,
            #                                            (loss.item() + np.log(256.) * (channel*img_size*img_size)) \
            #                                            / ((channel*img_size*img_size) * np.log(2.))))
            #print('BPD at iteration {}/{} is {}'.format(i, size, bpd))
            print('BPD2 at iteration {}/{} is {}'.format(i, size, bpd2))
            bpds_.append(bpd2)

    print('BPD average is {}'.format(np.mean(bpds_)))
    return np.mean(bpds_)

def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            x, log_pre = pre_process(x, False)
            z, log_dz_by_dx = model(x)
            loss, _, _ = loss_function(target_distribution, z, log_dz_by_dx, log_pre)
            total_loss += loss  # * x.size(0)
    return (total_loss / len(data_loader.dataset)).cpu().item()


def train_and_eval(flow, input_dim, epochs_to_learn, lr, train_loader, test_loader,
                   target_distribution, epoch, train_losses, test_losses, bpds, filename):
    """

    train&eval function

    :param flow: model to train
    :param input_dim:
    :param epochs_to_learn:
    :param lr: learning rate for optimizer
    :param train_loader: processed train data
    :param test_loader: processed test data
    :param target_distribution:
    :param epoch: current epoch
    :param train_losses: list of train losses previous epochs
    :param test_losses: list of test losses previous epochs
    :param bpds: list of bpds previous epochs
    :param filename: name for model savefile
    :return:
    """
    print('no of parameters is', sum(param.numel() for param in flow.parameters()))
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    print('Start epoch:', epoch, ' | Target epoch: ', epoch + epochs_to_learn - 1)
    exmp_imgs, _ = next(iter(train_loader))
    for e in range(epochs_to_learn):
        #print('Start epoch:', epoch, ' | Target epoch: ', epoch + epochs_to_learn-1)
        print('Starting epoch ', epoch, ' now.')
        #train(flow, train_loader, optimizer, target_distribution)
        bpds.append(train(flow, train_loader, optimizer, target_distribution))
        train_losses.append(eval_loss(flow, train_loader, target_distribution))
        test_losses.append(eval_loss(flow, test_loader, target_distribution))
        save_checkpoint(epoch, flow, train_losses, test_losses, bpds, filename)
        sample(flow, input_dim, target_dist=target_distribution, epoch=epoch)
        if e % 2 == 0:
            show_dist(flow, input_dim, target_distribution, exmp_imgs, epoch)
        epoch = epoch + 1
    return flow, train_losses, test_losses, bpds

if __name__ == '__main__':
    device = args.device
    print('Device is:', device)
    print('Dataset is:', dataset)
    import numpy as np

    # test_dataset(train_loader)
    # preprocess_test(train_loader)

    lr = args.lr
    epochs_to_learn = args.epochs_to_learn
    num_ex = args.num_ex  # 9, 16, 25, 36,....
    filename = args.filename
    model = args.model

    # nputs, classes = next(iter(train_loader))
    # print(nputs.size())
    # print(nputs[0].size(0))
    # input_dim = (nputs[0].size(0), nputs[0].size(1), nputs[0].size(2))

    input_dim = (channel, img_size, img_size)
    print("size of input data: " + str(input_dim))

    if model == 'NICE':
        flow = NICE(input_dim).to(device)

    elif model == 'RealNVP':
        flow = RealNVP(input_dim).to(device)

    elif model == 'RealNVP2':
        flow = RealNVP2(input_dim).to(device)

    flow, start_epoch, train_losses, test_losses, bpds = load_checkpoint(flow, filename)
    target_distribution = args.target_dist
    flow, train_losses, test_losses, bpds = train_and_eval(flow, input_dim, epochs_to_learn, lr, train_loader, test_loader,
                                                     target_distribution, start_epoch, train_losses, test_losses, bpds, filename)
    print('train losses are', train_losses)
    print('test losses are', test_losses)
    print('bpds are', bpds)

    # save_checkpoint(start_epoch + epochs_to_learn, flow, train_losses, test_losses)
    # sample(flow, input_dim, epoch=start_epoch, num_ex=num_ex)
    # sample(flow, input_dim)
    #exmp_imgs, _ = next(iter(train_loader))
    #for i in range(10):
    #    interpolate(flow, exmp_imgs[2*i], exmp_imgs[2*i+1])