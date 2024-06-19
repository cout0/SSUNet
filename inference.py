"""Model Inference

author: Masahiro Hayashi

This script allows users to make inference and visualize results.
"""
import os
import argparse
import torch
from torchvision import transforms
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from score import *
import time

from models import UNet, UNet_SNN, SNN_VGG, SpikingUnet, DMTSpikingUnet

model_idx = 0
model_name = ['UNet', 'UNet_SNN', 'SNN_VGG', 'SpikingUnet', 'DMTSpikingUnet']
loss_idx = 1
loss_name = ['Weighted_Cross_Entropy_Loss', 'DWCE_Loss']
g_epochs = 300

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predicitons'
    )
    parser.add_argument(
        '--model', type=str, default=f'{model_name[model_idx]}{g_epochs}.pt',
        help='model to use for inference'
    )
    parser.add_argument(
        '--visualize', action='store_true', default=True,
        help='visualize the inference result'
    )
    args = parser.parse_args()
    return args

def predict(image, model):
    """Make prediction on image"""
    mean = 0.495
    std = 0.173
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # transforms.Pad(22, padding_mode='reflect')
    ])
    im = image_transform(image)
    im = im.view(1, *im.shape)
    model.eval()
    start_time = time.time()
    y_pred = model(im)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f'Inference time for one pass: {inference_time:.4f} seconds')
    pred = torch.argmax(y_pred, dim=1)[0]
    return {'pred':pred, 'inft':inference_time}

def visualize(image, pred, label=None):
    """make visualization"""
    image_dict = {}
    n_plot = 2 if label is None else 3
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(1, n_plot, 1)
    plt.imshow(image)
    ax.set_title('Image')
    image_dict.update({'image.png': image})
    ax = fig.add_subplot(1, n_plot, 2)
    plt.imshow(pred)
    ax.set_title('Prediction')
    image_dict.update({'pred.png': pred})
    if n_plot > 2:
        ax = fig.add_subplot(1, n_plot, 3)
        plt.imshow(label)
        image_dict.update({'label.png': label})
        ax.set_title('Ground Truth')
    plt.show()
    plt.savefig(f'visualization/ISBI_2012_EM/{args.model[:-3]}_validation.png', bbox_inches='tight')


    for key, value in image_dict.items():
        plt.clf()
        plt.axis('off')
        plt.imshow(value)
        plt.savefig(f'./tmp/{key}', bbox_inches='tight')


if __name__ == '__main__':
    # start_time = time.time()
    args = parse_args()

    # load images and labels
    path = os.getcwd() + '/data/ISBI_2012_EM/test-volume.tif'
    images = io.imread(path)
    label_path = os.getcwd() + '/data/ISBI_2012_EM/test-labels.tif'
    labels = io.imread(label_path)
    # image = images[-1]
    # label = labels[-1]

    # load model
    checkpoint_path = os.getcwd() + f'/checkpoints/{args.model}'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = locals()[model_name[model_idx]](2)
    model.load_state_dict(checkpoint['model_state_dict'])

    # make inference
    pred = None
    total_time = 0.0
    rand = 0.0
    info = 0.0
    for idx in range(len(images)):
        pred_result = predict(images[idx], model)
        result = pred_result['pred']
        total_time += pred_result['inft']
        labels[idx] //= 255
        if idx == 0:
            pred = result
        rand += Vrand(result.numpy(), labels[idx])
        info += Vinfo(result.numpy(), labels[idx])

    rand /= len(labels)
    info /= len(labels)
    total_time /= len(labels)
    print(f'Average inference time for one pass: {total_time:.4f} seconds')

    if args.visualize:
        # crop images for visualization
        # dim = image.shape
        out_size = pred.shape[0]
        # cut = (dim[0] - out_size) // 2
        # image = image[cut:cut+out_size, cut:cut+out_size]
        # label = label[cut:cut+out_size, cut:cut+out_size]
        # visualize result
        visualize(images[0], pred, labels[0])

    # end_time = time.time()
    # duration = end_time - start_time
    # print("代码运行时长：", duration, "秒")
