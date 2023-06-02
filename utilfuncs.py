import os
import torch
import datetime

from torchvision import utils
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

from args import filename, batch_size, channel, img_size, device, target_dist, num_ex, savefolder, dataset, model
from NICE import NICE
from RealNVP import RealNVP
from RealNVP2 import RealNVP2
from utillayers import Preprocess

def show_imgs(imgs, epoch=None):
    nrow = int(len(imgs) ** 0.5)
    ncol = int(len(imgs) ** 0.5)
    #print("clean imgs")
    #print(imgs)
    imgs = utils.make_grid(imgs, nrow=nrow, normalize=True, scale_each=True, pad_value=0.1)
    #print("GRID IMGS?")
    #print(imgs)
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)))
    plt.axis('off')
    # plt.show()
    # save images
    if epoch is not None:
        ct = datetime.datetime.now()
        ct = ct.strftime('%d-%m-%Y_%H-%M-%S')
        plt.savefig(savefolder + 'generated_' + str(epoch) + '_' + ct)
    #plt.show()
    plt.close()

def show_imgs_(imgs, epoch=None):
    nrow = int(len(imgs) ** 0.5)
    ncols = int(len(imgs) ** 0.5)

    fig, axes = plt.subplots(nrow, ncols, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        # Umwandlung des aktuellen Tensors in ein numpy Array
        img = np.transpose(imgs[i], (1, 2, 0))  # (1, 28, 28) -> (28, 28, 1)
        # Anzeigen des aktuellen Bilds
        ax.imshow(img.squeeze(), cmap='gray')
        # Entfernen der Achsenbeschriftung
        ax.axis('off')

    plt.show()

def sample(model, input_dim, target_dist, epoch=None, num_ex=num_ex, z_=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if z_ == None:
        z = target_dist.sample([num_ex,
                                input_dim[0],
                                input_dim[1],
                                input_dim[2]])
    else:
        z = z_
    z = z.to(device)
    with torch.no_grad():
        model.eval()
        x = model.inverse(z)
        x = pre_process(x, True)

    if epoch is not None:
        return show_imgs(x, epoch)

    show_imgs(z)
    show_imgs(x)
    return


def load_checkpoint(model, filename=filename):
    if os.path.isfile(filename):
        print("checkpoint '{}' found".format(filename))
        checkpoint = torch.load(filename)
        start_epoch_ = checkpoint['epoch']+1
        print("Epochs trained already: '{}'".format(start_epoch_))
        model.load_state_dict(checkpoint['state_dict'])
        train_losses_ = checkpoint['train_losses']
        test_losses_ = checkpoint['test_losses']
        bpds_ = checkpoint['bpds']
        print("checkpoint loaded")


    else:
        print("there is no checkpoint '{}'".format(filename))
        start_epoch_ = 1
        train_losses_, test_losses_, bpds_ = [], [], []

    return model, start_epoch_, train_losses_, test_losses_, bpds_


def save_checkpoint(epochs, flow, train_losses, test_losses, bpds, filename=filename):
    state = {'epoch': epochs, 'state_dict': flow.state_dict(), 'train_losses': train_losses, 'test_losses': test_losses, 'bpds': bpds}
    if os.path.isfile(filename):
        print("checkpoints '{}' found".format(filename))
        torch.save(state, filename)
        print("model ran " + str(epochs) + " now")
        test = torch.load(filename)
        print("model ran " + str(test['epoch']) + " now (loaded test)")
        print("checkpoints saved at '{}'".format(filename))

    else:
        print("there is no checkpoint '{}'".format(filename))
        print("creating checkpoint '{}'".format(filename))
        torch.save(state, filename)

        backupname = 'backup.pt.tar'
        print("backup checkpoint '{}' just to be save".format(backupname))
        torch.save(state, backupname)


def test_dataset(data):
    print("testing dataset")
    fig, axes = plt.subplots(1, 10, figsize=(12, 3))

    for b in data:
        print(b[0].shape)
        print(torch.mean(b[0]))
        for i in range(10):
            axes[i].imshow(b[0][i].permute(1, 2, 0))
            # break
        break

    print("looks good?")
    # plt.waitforbuttonpress()


def preprocess_test(img):
    img = img.to(device)
    img += Uniform(0.0, 1/256).sample(img.size()).to(device)

    trans_img = 0.05+0.9*img
    trans_img = torch.logit(trans_img)

    #trans_img = torch.clip(trans_img, min=0, max=1)

    rest_img = torch.sigmoid(trans_img)
    rest_img = (rest_img-0.05)/0.9

    rest_img = torch.clip(rest_img, min=0, max=1)

    imgs = torch.cat([img.unsqueeze(0), trans_img.unsqueeze(0), rest_img.unsqueeze(0)], dim=0)
    show_imgs(imgs)
    return

def pre_process(data, reverse = False):

    if reverse:
        rest_img = torch.sigmoid(data)
        rest_img = (rest_img - 0.05) / 0.9
        #rest_img = torch.floor(rest_img*255)
        #print(rest_img)
        return rest_img

    #data *= 255 #NEW war ausgegraut, testen in int value (toTensor macht [0,255] -> [0,1])
    data += Uniform(0.0, 1).sample(data.size()).to(device)
    data = data/256
    #print(data.shape[1:])
    #log_jac_1 = -np.log(256) * np.prod(data.shape[1:])
    log_jac_1 = torch.full(data.shape, -np.log(256)).to(device)
    trans_data1 = 0.05+0.9*data
    trans_data = torch.logit(trans_data1)

    log_jac = log_jac_1 + np.log(0.9) + torch.log(1/(trans_data1*(-trans_data1+1)))
    return trans_data, log_jac

def show_imgs_interpolate(imgs, save):
    #print(len(imgs))
    imgs = utils.make_grid(imgs, nrow=len(imgs) // 3, normalize=True, scale_each=True, pad_value=0.1)
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(15, 15))
    #print(np_imgs.shape)
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)))
    plt.axis('off')
    # plt.show()
    # save images
    if save:
        ct = datetime.datetime.now()
        ct = ct.strftime('%d-%m-%Y_%H-%M-%S')
        plt.savefig('interpolation_' + ct)
    plt.show()


def interpolate(model, img1, img2, num_steps=10):
    """
    Inputs:
        model - trained model for interpolation
        img1, img2 - imgs to interpolate
        num_steps - steps in interpolation
    """
    imgs = torch.stack([img1, img2], dim=0).to('cuda')
    og_imgs = imgs
    #print("OG IMGS")
    #print(og_imgs)
    #show_imgs(og_imgs)
    model.eval()
    imgs, _ = pre_process(imgs, False) #NEU
    z, _ = model(imgs)
    #print("ZZZ")
    #print(z)
    #show_imgs(z)
    alpha = torch.linspace(0, 1, steps=num_steps, device=z.device).view(-1, 1, 1, 1)
    # print(alpha)
    interpolations = z[0:1] * alpha + z[1:2] * (1 - alpha)
    int22 = og_imgs[0:1] * alpha + og_imgs[1:2] * (1 - alpha)
    #print("OG IMGS INTERPOLATED")
    #print(int22)
    #print(int22.shape)
    #show_imgs(int22)
    # print(interpolations)

    # show_imgs_interpolate(interpolations, False)

    with torch.no_grad():
        # interp_imgs = sample(model, (1,28,28), z_=interpolations)
        interp_imgs = model.inverse(interpolations)
        interp_imgs = pre_process(interp_imgs, True)
        #show_imgs(torch.cat([interpolations, interp_imgs, int22]))
        show_imgs_interpolate(torch.cat([interpolations, interp_imgs, int22], dim=0)
                              , True)

def interpolate2(model, img1, img2, img3, img4, num_steps=5):
    """
    Inputs:
        model - trained model for interpolation
        img1, img2, img3, img4 - imgs to interpolate
        num_steps - steps in interpolation
    """
    A = img1
    B = img2
    C = img3
    D = img4

    imgs = torch.stack([A, B, C, D], dim=0).to('cuda')
    model.eval()
    imgs, _ = pre_process(imgs, False)
    z, _ = model(imgs)
    #print(z.shape)
    #print(z[0].shape)

    A = z[0]
    B = z[1]
    C = z[2]
    D = z[3]

    size = (num_steps + 2, num_steps + 2)

    interpolated_tensors = []

    for i in range(size[0]):
        for j in range(size[1]):
            t = torch.tensor([i / (size[0] - 1), j / (size[1] - 1)])
            print(t)

            horizontal_interpolation_AB = (1 - t[1]) * A + t[1] * B
            horizontal_interpolation_CD = (1 - t[1]) * C + t[1] * D

            interpolated_tensor = (1 - t[0]) * horizontal_interpolation_AB + t[0] * horizontal_interpolation_CD

            interpolated_tensors.append(interpolated_tensor)

    stacked_tensor = torch.stack(interpolated_tensors, dim=0)

    with torch.no_grad():
        # interp_imgs = sample(model, (1,28,28), z_=interpolations)
        interp_imgs = model.inverse(stacked_tensor)
        interp_imgs = pre_process(interp_imgs, True)

    show_imgs(interp_imgs, epoch=99999)
    return stacked_tensor

def test_model(model, img):
    """

    :param model: model to test transformations
    :param img: example img for test
    :return:
    """
    img = img.to('cuda')
    img = img.unsqueeze(0)
    print("test_model")
    print(img.size())
    with torch.no_grad():
        model.eval()
        z, _ = model(img)
        # print(z)
        # print(z)
        z2 = model.inverse(z)

    imgs = torch.cat([img, z, z2], dim=0)
    # print(imgs.size())
    #show_imgs_interpolate(imgs, False)
    img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    z = z.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    z2 = z2.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    # print(x.shape)

    # def min_max_normalize(x):
    #    return (x - x.min()) / (x.max() - x.min())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes = axes.ravel()

    axes[0].imshow(img)
    axes[1].imshow(z)
    axes[2].imshow(z2)
    plt.show()

def prep_dataset(dataset):
    """

    :param dataset: dataset to prepare for model
    :return:
    """
    if dataset == 'mnist':
        transform_ = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x*255 #NEW TEST
        ])

        train_dataset = datasets.MNIST('data', train=True, download=False, transform=transform_)
        test_dataset = datasets.MNIST('data', train=False, download=False, transform=transform_)

        #print(len(train_dataset))
        #print(len(test_dataset))

        used_train, _ = torch.utils.data.random_split(train_dataset, [35000, 25000])
        used_test, _ = torch.utils.data.random_split(test_dataset, [5000, 5000])


        test_loader = DataLoader(used_test, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(used_train, batch_size=batch_size, shuffle=True)

    elif dataset == 'celebA':
        transform_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            lambda x: x * 255,
            #transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                    std=[0.5, 0.5, 0.5]),
        ])

        train_dataset = datasets.CelebA(root='data', split='train', download=False, transform=transform_)
        test_dataset = datasets.CelebA(root='data', split='test', download=False, transform=transform_)

        #print(len(train_dataset))
        #print(len(test_dataset))

        used_train, _ = torch.utils.data.random_split(train_dataset, [5000, 157770])
        used_test, _ = torch.utils.data.random_split(test_dataset, [2000, 17962])

        test_loader = DataLoader(used_test, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(used_train, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def show_dist(flow, input_dim, target_dist_, data, epoch=None):
    trgt_dist = target_dist_

    with torch.no_grad():
        x = data
        #print(x.mean())
        x = x.to(device)
        #print(x.mean())
        x1, _ = pre_process(x, False)
        z, _ = flow(x1)
        trgt = trgt_dist.sample([batch_size, input_dim[0], input_dim[1], input_dim[2]])
        #print(x.shape)
        #print(z.shape)
        #print(trgt.shape)
        df = pd.DataFrame({
            'x': torch.flatten(x).cpu(),
            'z': torch.flatten(z).cpu(),
            'trgt': torch.flatten(trgt).cpu(),
        })
        df.hist(column=['x'], bins=50)
        df.hist(column=['z', 'trgt'], bins=50, figsize=(10, 8), sharex=True, range=(-10, 10), layout=(2, 1))
        #print(x.mean())
        #print(torch.flatten(x).mean())
        if isinstance(epoch, int):
            ct = datetime.datetime.now()
            ct = ct.strftime('%d-%m-%Y_%H-%M-%S')
            plt.savefig(savefolder + 'histogram_' + str(epoch) + '_' + ct)
        plt.close()
        #ax = df.plot.hist(column=['z'], by="x", figsize=(10, 8))
        #fig = ax.get_figure()
        #fig.savefig('test.pdf')

    return

def show_distr(data):
    df = pd.DataFrame({
        'data': torch.flatten(data),
    })
    df.hist(column=['data'], bins=50)
    ct = datetime.datetime.now()
    ct = ct.strftime('%d-%m-%Y_%H-%M-%S')
    plt.savefig('dataimgs/generated_'+ct)
    plt.close()

def show_ex(data):
    nrow = 5
    ncol = 2
    data = data[0:10]
    imgs = utils.make_grid(data, nrow=nrow, ncol=ncol, pad_value=0.1)
    #print(len(data[0:10]))
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(nrow, ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)))
    plt.axis('off')
    # plt.show()
    # save images
    ct = datetime.datetime.now()
    ct = ct.strftime('%d-%m-%Y_%H-%M-%S')
    plt.savefig('dataimgs/generated_'+ct)
    plt.close()


if __name__ == '__main__':

    """
    main used for testing specific things + creating data to showcase
    """
    #train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_loader, test_loader = prep_dataset(dataset)

    input_dim = (channel, img_size, img_size)
    model = model

    if model == 'NICE':
        flow = NICE(input_dim).to(device)

    elif model == 'RealNVP':
        flow = RealNVP(input_dim).to(device)

    elif model == 'RealNVP2':
        flow = RealNVP2(input_dim).to(device)

    #print(flow)

    flow, start_epoch, train_losses, test_losses, bpds = load_checkpoint(flow, filename)
    print('no of parameters is', sum(param.numel() for param in flow.parameters()))

    #print(bpds)

    exmp_imgs, _ = next(iter(train_loader))

    #print(exmp_imgs[0])

    #print(exmp_imgs.max())
    #print(exmp_imgs.mean())
    #show_dist(flow, input_dim, target_dist, exmp_imgs)

    #torch.set_printoptions(profile="full")
    #interpolate(flow, exmp_imgs[0], exmp_imgs[1])
    #torch.set_printoptions(profile="default")
    interpolate2(flow, exmp_imgs[0], exmp_imgs[1], exmp_imgs[2], exmp_imgs[3], num_steps=5)
    #show_distr(exmp_imgs)
    #show_ex(exmp_imgs)
    #test_model(flow, exmp_imgs[0])
    #torch.set_printoptions(profile="full")
    #show_imgs(exmp_imgs)
    #show_imgs(exmp_imgs*255.)
    #torch.set_printoptions(profile="default")

    #preprocess_test(exmp_imgs[0])
    #for i in range(10):
    #    interpolate(flow, exmp_imgs[i], exmp_imgs[i+2])
    #sample(flow, input_dim, target_dist=target_dist)