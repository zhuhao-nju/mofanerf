import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2

modelFolder = 'trained_model/'

# load model
from defineHourglass_512_gray_skip import *
class RelightModule():
    def __init__(self):
        my_network = HourglassNet()
        my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_03.t7')))
        my_network.cuda()
        my_network.train(False)
        self.RelMo = my_network
        self.srcSH = np.load("fcspLight.npy")
        # self.device = device

    def trans_get_sh(self, RGB):  # returen sh of src_img, and tar_sh trans result
        Lab = cv2.cvtColor(RGB, cv2.COLOR_RGB2LAB)
        inputL = Lab[:,:,0]
        sh = self.srcSH
        
        # inputL = inputL.transpose(0, 1)
        # inputL = inputL[0,0,...]
        inputL = inputL[None, None, ...]
        inputL = torch.from_numpy(inputL / 255.).float().cuda()
        
        
        sh = Variable(torch.from_numpy(sh).cuda())
        outputImg, outputSH = self.RelMo(inputL, sh, 0)
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1, 2, 0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg * 255.0).astype(np.uint8)
        if Lab is not None:
            Lab[:, :, 0] = outputImg
            resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB)
            outputImg = resultLab
            # outputImg = cv2.resize(resultLab, (col, row))
        # plt.imshow(outputImg)
        # plt.show()
        return outputImg, outputSH.detach().cpu().numpy()