import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        # self.decode_0=nn.Conv2d(80,64,kernel_size=3,padding=1,padding_mode='reflect')
        self.decode_1=nn.Conv2d(64,48,kernel_size=3,padding=1,padding_mode='reflect')
        self.decode_2=nn.Conv2d(48,32,kernel_size=3,padding=1,padding_mode='reflect')
        self.decode_3=nn.Conv2d(32,16,kernel_size=3,padding=1,padding_mode='reflect')
        self.decode_4=nn.Conv2d(16,1,kernel_size=3,padding=1,padding_mode='reflect')

        self.relu=F.relu
        # self.relu=nn.PReLU()
    
    def forward(self,features):
        # features=self.decode_0(features)
        # features=self.relu(features)

        features=self.decode_1(features)
        features=self.relu(features)

        features=self.decode_2(features)
        features=self.relu(features)

        features=self.decode_3(features)
        features=self.relu(features)

        features=self.decode_4(features)
        features=self.relu(features)

        return features 

class Encoder(nn.Module):
    def __init__(self,k,d,in_channels=None):
        super(Encoder, self).__init__()

        self.k = k
        self.d = d
        self.in_channels = in_channels
        self.ini_channels = 16

        self.conv2d_x=nn.Conv2d(1,16,kernel_size=3,padding=1,padding_mode='reflect')

        self.filters1x = nn.ParameterList(
            [nn.Parameter(torch.randn(self.k, self.ini_channels + self.k * i, 3, 3), requires_grad=True) for i in
             range(self.d)])


        self.b1x = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.ini_channels + self.k + self.k * i, 1, 1), requires_grad=True) for i in
             range(self.d)])
        
        
        self.bn1x = [nn.BatchNorm2d(self.ini_channels+(i+1)*self.k, affine=True).cuda() for i in
                    range(self.d)]

        self.c1x = nn.ParameterList(
            [nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True) for i in range(self.d)])

        self.relu=F.relu
        # self.relu=nn.PReLU()

        # Initialization
        for i in range(self.d):
            self.filters1x[i].data = .1 / np.sqrt((self.ini_channels + self.k * i) * 9) * self.filters1x[i].data


    def MSD_CSC_ISTA_Block(self, input, k, d, filters, b, bn, c, dilation_cycle,unfolding):
        features = input
        for i in range(d):
            f1 = F.conv2d(features, filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                          dilation=(i % dilation_cycle) + 1)
            f2 = torch.cat((features, f1), dim=1)
            del f1
            f3 = c[i] * f2 + b[i]
            del f2
            features=F.relu(bn[i](f3))
            del f3
        
        # backward
        for loop in range(unfolding):
            for i in range(d - 1):
                f1 = F.conv_transpose2d(features[-1 - i][ -k:, :, :], filters[-1 - i], stride=1,
                                        padding=((-1 - i + d) % dilation_cycle) + 1,
                                        dilation=((-1 - i + d) % dilation_cycle) + 1)
                features[-2 - i] = f1 + features[-1 - i][ 0:-k, :, :]
            # forward
            #print("forward")
            for i in range(d):
                #print(i)
                f1 = F.conv_transpose2d(features[i + 1][ -k:, :, :], filters[i], stride=1,
                                        padding=(i % dilation_cycle) + 1, dilation=(i % dilation_cycle) + 1)
                f2 = features[i + 1][ 0:-k, :, :] + f1
                del f1
                f3 = F.conv2d(f2, filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                              dilation=(i % dilation_cycle) + 1)
                f4 = torch.cat((f2, f3), dim=1)  ###
                del f2,f3
                f5 = F.conv2d(features[i], filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                              dilation=(i % dilation_cycle) + 1)
                f6 = torch.cat((features[i], f5), dim=1)  ###
                f7 = features[i + 1] - c[i] * (f4 - f6) + b[i]
                del f4,f6
                features[i + 1] = F.relu(bn[i](f7))

        return features

    def MSD_CSC_FISTA_Block(self,input, k, d, filters, b, bn, c, dilation_cycle, unfolding):
        t = 1
        t_prv = t
        # Encoding
        features = []
        features.append(input)
        for i in range(d):
            f1 = F.conv2d(features[-1], filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                          dilation=(i % dilation_cycle) + 1)
            f2 = torch.cat((features[-1], f1), dim=1)
            del f1
            f3 = c[i] * f2 + b[i]
            del f2
            features.append(self.relu(bn[i](f3)))
        feature_prv = features[-1]

        for loop in range(unfolding):

            t_prv = t
            t = float((1 + np.sqrt(1 + 4 * t_prv ** 2)) / 2)

            Z = features[-1] + (t_prv - 1) / t * (features[-1] - feature_prv)
            feature_prv = features[-1]
            features[-1] = Z

            # backward
            for i in range(d - 1):
                f1 = F.conv_transpose2d(features[-1 - i][:, -k:, :, :], filters[-1 - i], stride=1,
                                        padding=((-1 - i + d) % dilation_cycle) + 1,
                                        dilation=((-1 - i + d) % dilation_cycle) + 1)
                features[-2 - i] = f1 + features[-1 - i][:, 0:-k, :, :]

            # forward
            for i in range(d):
                #print(i)
                f1 = F.conv_transpose2d(features[i + 1][:, -k:, :, :], filters[i], stride=1,
                                        padding=(i % dilation_cycle) + 1, dilation=(i % dilation_cycle) + 1)
                f2 = features[i + 1][:, 0:-k, :, :] + f1
                del f1
                f3 = F.conv2d(f2, filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                              dilation=(i % dilation_cycle) + 1)
                f4 = torch.cat((f2, f3), dim=1)  ###
                del f2, f3
                f5 = F.conv2d(features[i], filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                              dilation=(i % dilation_cycle) + 1)
                f6 = torch.cat((features[i], f5), dim=1)  ###
                f7 = features[i + 1] - c[i] * (f4 - f6) + b[i]
                del f4, f6
                features[i + 1] = self.relu(bn[i](f7))
        return features[-1]

    def forward(self,x):
        x = self.conv2d_x(x)
        # x = self.bn0x(x)
        x = self.relu(x)
        x = self.MSD_CSC_FISTA_Block(x,self.k,self.d,self.filters1x,self.b1x,self.bn1x,self.c1x,6,2)
        return x


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet,self).__init__()
        self.encoder=Encoder(16,3,1)
        self.decoder=Decoder()
    
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
