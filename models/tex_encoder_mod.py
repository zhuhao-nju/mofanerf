import torch
import torch.nn as nn

from models.utils_texEncoder import *


class EnDeUVmap(nn.Module):
    def __init__(self, uvCodesLen=256):
        super(EnDeUVmap, self).__init__()
        self.encoder = Encoder(1, uvCodesLen=uvCodesLen)  # 1 means process input once
        # self.decoder = Decoder(globalwarp=False, templateres=80)
        # self.volsampler = VolSampler()

    def forward(self, uvMap, lossList):
        encoding, losses = self.encoder(uvMap, lossList)
        # = encOur["encoding"]
        # decOut = self.decoder(encoding, lossList)['template']
        # return decOut
        return encoding, losses


class Encoder(torch.nn.Module):
    def __init__(self, ninputs, tied=False, uvCodesLen=256):
        super(Encoder, self).__init__()

        self.ninputs = ninputs
        self.tied = tied

        # self.down1 = nn.ModuleList([nn.Sequential(
        #         nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
        #         nn.Conv2d(64, 64, 4, 2, 1), nn.LeakyReLU(0.2),
        #         nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
        #         nn.Conv2d(128, 128, 4, 2, 1), nn.LeakyReLU(0.2),
        #         nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
        #         nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
        #         # nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
        #         nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2))
        #         for i in range(1 if self.tied else self.ninputs)])
        self.down1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            # nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2))
            for i in range(1 if self.tied else self.ninputs)])
        self.down2 = nn.Sequential(
            nn.Linear(256 * self.ninputs * 4 * 4, 512), nn.LeakyReLU(0.2))
        self.height, self.width = 512, 512
        ypad = ((self.height + 127) // 128) * 128 - self.height
        xpad = ((self.width + 127) // 128) * 128 - self.width
        self.pad = nn.ZeroPad2d((xpad // 2, xpad - xpad // 2, ypad // 2, ypad - ypad // 2))
        self.mu = nn.Linear(512, uvCodesLen)
        self.logstd = nn.Linear(512, uvCodesLen)

        for i in range(1 if self.tied else self.ninputs):
            initseq(self.down1[i])  # utils_EnDe.
        initseq(self.down2)
        initmod(self.mu)
        initmod(self.logstd)

        self.decoding = nn.Sequential(
            nn.Linear(uvCodesLen, uvCodesLen),
            nn.LeakyReLU(0.1),
            nn.Linear(uvCodesLen, uvCodesLen),
            nn.LeakyReLU(0.1),
            nn.Linear(uvCodesLen, uvCodesLen),
            nn.LeakyReLU(0.1),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(0.1)
        )
        for m in self.decoding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.last.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, losslist=[]):  # [16,9,512,334]

        # return {"encoding": torch.rand(x.shape[0], 256), "losses": {}+}
        x = self.pad(x)  # [16,9,512,384]
        _, c, h, w = x.shape
        # ; here three input image x which dim1=9, is spiltted into three image， and view as one dimention
        x = [self.down1[0 if self.tied else i](x[:, :, :, :]).view(-1, 256 * 4 * 4) for i in range(self.ninputs)]
        x = torch.cat(x, dim=1)  # cat three image together to get x  [16, C*H*W]
        x = self.down2(x)
        # mu = self.mu(x)
        # mu, logstd = self.mu(x) * 0.1, self.logstd(x) * 0.01
        # if self.training:
        #     z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)#ndn -- normalization
        # else:
        #     z = mu
        z = self.mu(x)
        out = self.decoding(z)
        losses = {}
        # if "loss_kldiv" in losslist:
        #     losses["loss_kldiv"] = torch.mean(-0.5 - logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)

        return out, losses
# class Decoder(nn.Module):
#     def __init__(self, input_ch, W):
#         super(Decoder, self).__init__()
#         self.decoding = nn.Sequential(nn.Linear(input_ch,))
