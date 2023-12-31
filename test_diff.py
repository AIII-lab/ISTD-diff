# -*-coding:utf-8-*-
import torch
import torch.utils.data as Data
from argparse import ArgumentParser
from tqdm import tqdm
import os
import numpy as np
from dataset import Sirst_Dataset
from ISTDdiff import ISTDdiff
from metrics import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA, ROCMetric, mIoU
import cv2
import logging
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def parse_args():
    # Setting parameters
    parser = ArgumentParser(description='Implement of ISTD-diff model:')
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--load_name', type=str, default='./model/NUAA-SIRST-model.pkl', help='blocks per layer')
    args = parser.parse_args()
    return args

class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.noise_steps_sample = 500
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].cuda()
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].cuda()
        noise = torch.randn(size=x.shape).cuda()
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, data):
        n = data.shape[0]
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).cuda()
            for i in reversed(range(1, self.noise_steps_sample)):
                t = torch.tensor([i] * x.shape[0])
                # print(x.shape, t.shape)

                # print(f"完成第{i}步")
                predicted_noise = model(x, t.cuda(), data)
                alpha = self.alpha[t][:, None, None, None].cuda()
                alpha_hat = self.alpha_hat[t][:, None, None, None].cuda()
                beta = self.beta[t][:, None, None, None].cuda()
                if i > 1:
                    noise = torch.randn(size=x.shape).cuda()
                else:
                    noise = torch.zeros_like(x).cuda()
                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                # x = 1 / torch.sqrt(alpha) * (x - predicted_noise)
                # x = predicted_noise
                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                x = torch.sqrt(alpha_hat)*(1-beta) / (torch.sqrt(1 - alpha_hat)) * predicted_noise + torch.sqrt(alpha)*(1-alpha_hat) / (torch.sqrt(1 - alpha_hat)) * x + torch.sqrt(beta) * noise

        # model.train()
        # x = x.clip(0, 1)
        # x = (x * 255)
        return x

def save_image(label, output, name, result_path):
    output = output.detach().numpy()
    label = label.detach().numpy()
    output = (output > 0).astype('uint8')
    label = (label > 0).astype('uint8')
    output = output[0,0,:,:]*255
    label = label[0,0,:,:]*255
    output = np.array([output, output, output])
    label = np.array([label, label, label])
    output = np.transpose(output, [1,2,0])
    label = np.transpose(label, [1,2,0])
    save_path = os.path.join(result_path, name)
    cv2.imwrite(save_path, output)
    # plt.subplot(121), plt.imshow(label), plt.title('Label')
    # plt.subplot(122), plt.imshow(output), plt.title('Output')
    # plt.savefig(save_path, bbox_inches='tight')

class Trainer(object):
    def __init__(self, args):
        self.args = args
        valset = Sirst_Dataset(args, mode='test')
        self.val_data_loader = Data.DataLoader(valset, batch_size=1)
        print(len(self.val_data_loader))
        channels = [8, 16, 32]
        self.net = ISTDdiff(channels)
        self.diffusion = Diffusion(noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=args.base_size, device="cuda")
        save_point = torch.load(self.args.load_name)
        model_param = save_point['ISNet']
        model_dict = {}
        for k1, k2 in zip(self.net.state_dict(), model_param):
            model_dict[k1] = model_param[k2]
        self.net.load_state_dict(model_dict)
        device = torch.device("cuda")
        self.net.to(device)

        self.result_path = os.path.join('/'.join((args.load_name.split('/')[:-2])), 'result')
        print(self.result_path)
        if(not os.path.exists(self.result_path)):
            os.mkdir(self.result_path)

        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)

        self.best_iou = 0
        self.best_nIoU = 0
        self.best_FA = 1000000000000000
        self.best_PD = 0

        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10)
        self.mIoU = mIoU(1)

    def validation(self):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        self.net.eval()
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels, save_name) in enumerate(tbar):
            with torch.no_grad():
                output = self.diffusion.sample(self.net, data.cuda())
                labels = labels[:,0:1,:,:].cpu()
                output = output.cpu()
                save_image(labels, output, save_name[0][:-4]+''+save_name[0][-4:], self.result_path)

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.ROC.update(output, labels)
            self.PD_FA.update(output, labels)
            FA, PD = self.PD_FA.get(len(self.val_data_loader))
            _, mean_IOU = self.mIoU.get()
            _, IoU = self.iou_metric.get()
            print(save_name)
            print("Iou:",end='')
            print(IoU)
            _, nIoU = self.nIoU_metric.get()
            print("nIou:",end='')
            print(nIoU)

        if FA[0]*1000000 < self.best_FA:
            self.best_FA = FA[0]*1000000
        if PD[0] > self.best_PD:
            self.best_PD = PD[0]
        print('miou', mean_IOU)
        print('IoU', IoU)
        print('nIoU', nIoU)
        print('FA', FA[0] * 1000000)
        print('PD', PD[0])

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    trainer = Trainer(args)
    trainer.validation()

