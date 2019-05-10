import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.autograd import Variable
import torch
import pdb

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        
        output = (cos_theta, phi_theta)
        return output # size=(B,Classnum,2)

class NormLoss(nn.Module):
    def __init__(self, norm_weight):
        super(NormLoss, self).__init__()
        self.norm_weight = norm_weight

    def forward(self, input, xlen, target):
        cos_theta = input
        target = target.view(-1,1)
        xlen = xlen.view(-1,1)
        output = cos_theta
        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        bs = input.size(0)
        classnum = input.size(1)
        

        if self.norm_weight:
          gather_norm = torch.zeros(classnum, bs).cuda()
          gather_norm.scatter_(0, target.reshape(1,-1), xlen.reshape(1,-1))
          gather_num = torch.sum(gather_norm>0,1).float()
          average_norm = torch.sum(gather_norm,1)/gather_num
          inverse_norm = 1./average_norm 
          sample_weight = torch.zeros(bs,1)
          sample_weight = torch.gather(inverse_norm.reshape(-1,1), 0, target)
          #sample_weight = F.softmax(sample_weight)
          sample_weight = sample_weight / torch.sum(sample_weight)
          sample_weight = sample_weight.detach()
          loss = -torch.sum(sample_weight.view(-1)*logpt)
        else:
          loss = -logpt.mean()
          gather_norm = torch.zeros(classnum, bs).cuda()
          gather_norm.scatter_(0, target.reshape(1,-1), xlen.reshape(1,-1))
          gather_num = torch.sum(gather_norm>0,1).float()
          average_norm = torch.sum(gather_norm,1)/gather_num
          inverse_norm = 1./average_norm 
        return loss, torch.sum(gather_norm,1), gather_num


class NormLinear(nn.Module):
    def __init__(self, in_features, out_features, radius=None):
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.radius = radius
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.iter = 0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

        
    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features
        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        x_norm = x.renorm(2,0,1e-5).mul(1e5)
        bs = x.size(0)
        feature_dim = self.in_features
        classnum = self.out_features
        if self.radius:
          x = x_norm*self.radius
       
        cos_theta = x.mm(ww) # size=(B,Classnum)
        output = (cos_theta)
        
        return output # size=(B,Classnum,2)
    
class GaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, radius):
        super(GaussianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.radius = radius
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features
        ww = w.renorm(2,1,1e-5).mul(1e5) 
        class_num = self.out_features
        bs = x.size(0)
        x_square = x.pow(2).sum(1).reshape(-1,1).repeat(1,class_num) # size=B
        xw_dot = x.mm(ww)
        w_square = ww.pow(2).sum(0).reshape(1,-1).repeat(bs,1)
        output = -0.5*(x_square-2*xw_dot+w_square)
        pdb.set_trace()
        return output 
    
class AngleLoss(nn.Module):
    def __init__(self,gamma=0.00003,power=5,LambdaMin=5, LambdaMax=1000):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = LambdaMin
        self.LambdaMax = LambdaMax
        self.lamb = 1500.0
        self.power = power

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+self.gamma*self.it )**self.power)
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        #pt = Variable(logpt.data.exp())

        #loss = -1 * (1-pt)**self.gamma * logpt
        loss = -logpt.mean()

        return loss, self.lamb


class Net(nn.Module):
    def __init__(self, classnum,feature_dim=2, head='softmax', radius=None, sample_feat=False):
        super(Net, self).__init__()
        self.relu = nn.PReLU()
        self.radius = radius
        self.feature_dim = feature_dim
        self.sample_feat = sample_feat
        # self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        #self.ip1 = nn.Linear(128*3*3, feature_dim)
        self.mean = nn.Linear(128*3*3, feature_dim)
        self.var = nn.Linear(128*3*3, feature_dim)
        self.var_relu = nn.ReLU(512)

        if head == 'softmax':
            self.ip2 = NormLinear(feature_dim, classnum, self.radius)
        if head == 'a-softmax':
            self.ip2 = AngleLinear(feature_dim, classnum)
        if head == 'gaussian':
            self.ip2 = GaussianLinear(feature_dim, classnum, self.radius)
      

    def forward(self, x):
        bs = x.size(0)
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*3*3)
        mean = self.mean(x)
        mean = mean.renorm(2,0, 1e-5).mul(1e5)
        if self.sample_feat:
          var = self.var_relu(self.var(x))
          eps = torch.randn([bs, self.feature_dim]).cuda()
          ip1 = eps * var + mean
        else:
          var = torch.zeros([bs, self.feature_dim])
          ip1 = mean
        ip2 = self.ip2(ip1)
        return ip1, ip2, var
