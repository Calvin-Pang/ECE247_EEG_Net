import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.F1 = 8
        self.F2 = 16
        self.D = 2
        
        # Conv2d(in,out,kernel,stride,padding,bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        ) 
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (22, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        ) # 16x1x1001
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False), 
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.PReLU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout(0.5)
        ) # 16x1x15
        
        self.classifier = nn.Linear(16*15, 4, bias=True)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)
        
        x = x.view(-1, 16*15)
        x = self.classifier(x)
        return x

class Multi_EEGNet(nn.Module):
    def __init__(self):
        super(Multi_EEGNet, self).__init__()

        self.F1 = 8
        self.F2 = 16
        self.D = 2
        
        # Conv2d(in,out,kernel,stride,padding,bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        ) 
        init.kaiming_uniform_(self.conv1[0].weight, mode='fan_in', nonlinearity='relu')

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (22, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        ) # 16x1x250
        init.kaiming_uniform_(self.conv2[0].weight, mode='fan_in', nonlinearity='relu')

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False), # 16x1x251
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(self.F2),
            nn.PReLU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout(0.5)
        ) # 16x1x15
        init.kaiming_uniform_(self.conv3[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 8), padding=(0, 4), groups=self.D*self.F1, bias=False), # 16x1x251
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(self.F2), 
            nn.PReLU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout(0.5)
        )
        init.kaiming_uniform_(self.conv4[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv4[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 32), padding=(0, 16), groups=self.D*self.F1, bias=False), # 16x1x251
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(self.F2), 
            nn.PReLU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout(0.5)
        )
        init.kaiming_uniform_(self.conv5[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv5[1].weight, mode='fan_in', nonlinearity='relu')
        
        self.conv6 = nn.Conv2d(48, 16, (1,1))
        init.kaiming_uniform_(self.conv6.weight, mode='fan_in', nonlinearity='relu')

        self.classifier = nn.Linear(16*15, 4, bias=True)
        init.kaiming_uniform_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.conv3(x)
        x2 = self.conv4(x)
        x3 = self.conv5(x)
        
        x_con = torch.cat((x3, x1, x2), dim = 1)
        x_con = self.conv6(x_con)
        x_con = x_con.view(-1, 16*15)
        x_con = self.classifier(x_con)
        return x_con
    

class ShallowConvNet(nn.Module):
    def __init__(self):
        super(ShallowConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 40, (1, 13), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (22, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(40*137, 4, bias=True)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Bn1(x)
        x = x ** 2
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, 40*137)
        x = self.classifier(x)
        
        return x
    

class HR_EEGNet(nn.Module):
    def __init__(self):
        super(HR_EEGNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        ) 
        init.kaiming_uniform_(self.conv1[0].weight, mode='fan_in', nonlinearity='relu')

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, (22, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        ) # 16x1x250
        init.kaiming_uniform_(self.conv2[0].weight, mode='fan_in', nonlinearity='relu')
        
        self.conv3a = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False), # 16x1x251
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(16),
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout(0.5)
        ) # 16x1x15
        init.kaiming_uniform_(self.conv3a[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3a[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv3b = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), padding=(0, 4), groups=16, bias=False), # 16x1x251
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        ) # 16x1x62
        init.kaiming_uniform_(self.conv3b[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3b[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv4b = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), padding=(0, 4), groups=16, bias=False), # 16x1x32
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x32
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        ) # 16x1x15
        init.kaiming_uniform_(self.conv4b[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv4b[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv3c = nn.Sequential(
            nn.Conv2d(16, 16, (1, 4), padding=(0, 2), groups=16, bias=False), # 16x1x251
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(0.5)
        ) # 16x1x125
        init.kaiming_uniform_(self.conv3c[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3c[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv4c = nn.Sequential(
            nn.Conv2d(16, 16, (1, 4), padding=(0, 1), groups=16, bias=False), # 16x1x125
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x125
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(0.5)
        ) # 16x1x62
        init.kaiming_uniform_(self.conv4c[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv4c[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv5c = nn.Sequential(
            nn.Conv2d(16, 16, (1, 4), padding=(0, 2), groups=16, bias=False), # 16x1x62
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x62
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        ) # 16x1x15
        init.kaiming_uniform_(self.conv5c[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv5c[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv_32_16_b = nn.Conv2d(32, 16, (1,1))
        init.kaiming_uniform_(self.conv_32_16_b.weight, mode='fan_in', nonlinearity='relu')

        self.conv_32_16_c = nn.Conv2d(32, 16, (1,1))
        init.kaiming_uniform_(self.conv_32_16_c.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv_all = nn.Conv2d(48, 16, (1,1))
        init.kaiming_uniform_(self.conv_all.weight, mode='fan_in', nonlinearity='relu')

        self.classifier = nn.Linear(16*15, 4, bias=True)
        init.kaiming_uniform_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)

        x_a = self.conv3a(x)

        x_b = self.conv3b(x)
        x_c = self.conv3c(x)
        x_c = self.conv4c(x_c)

        x_b = torch.cat((x_b, x_c), dim = 1)
        x_b = self.conv_32_16_b(x_b)

        x_c = torch.cat((x_c, x_b), dim = 1)
        x_c = self.conv_32_16_c(x_c)

        x_b = self.conv4b(x_b)
        x_c = self.conv5c(x_c)

        x_con = torch.cat((x_a, x_b, x_c), dim = 1)

        x_con = self.conv_all(x_con)
        x_con = x_con.view(-1, 16*15)
        x_con = self.classifier(x_con)
        return x_con
    

class MSFNet(nn.Module):
    def __init__(self, drop = 0.4) :
        super(MSFNet, self).__init__()

        self.conv1a = nn.Sequential(
            nn.Conv2d(1, 8, (1, 128), padding=(0, 64), bias=False),
            nn.BatchNorm2d(8)
        ) 
        init.kaiming_uniform_(self.conv1a[0].weight, mode='fan_in', nonlinearity='relu')

        self.conv1b = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        ) 
        init.kaiming_uniform_(self.conv1b[0].weight, mode='fan_in', nonlinearity='relu')

        self.conv1c = nn.Sequential(
            nn.Conv2d(1, 8, (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(8)
        ) 
        init.kaiming_uniform_(self.conv1c[0].weight, mode='fan_in', nonlinearity='relu')

        self.conv_24_8 = nn.Conv2d(24, 8, (1,1))
        init.kaiming_uniform_(self.conv_24_8.weight, mode='fan_in', nonlinearity='relu')

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, (22, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop)
        ) # 16x1x250
        init.kaiming_uniform_(self.conv2[0].weight, mode='fan_in', nonlinearity='relu')

        
        self.conv3a = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False), # 16x1x251
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(16),
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout(drop)
        ) # 16x1x15
        init.kaiming_uniform_(self.conv3a[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3a[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv3b = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), padding=(0, 4), groups=16, bias=False), # 16x1x251
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop)
        ) # 16x1x62
        init.kaiming_uniform_(self.conv3b[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3b[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv4b = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), padding=(0, 4), groups=16, bias=False), # 16x1x32
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x32
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop)
        ) # 16x1x15
        init.kaiming_uniform_(self.conv4b[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv4b[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv3c = nn.Sequential(
            nn.Conv2d(16, 16, (1, 4), padding=(0, 2), groups=16, bias=False), # 16x1x251
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x251
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(drop)
        ) # 16x1x125
        init.kaiming_uniform_(self.conv3c[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3c[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv4c = nn.Sequential(
            nn.Conv2d(16, 16, (1, 4), padding=(0, 1), groups=16, bias=False), # 16x1x125
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x125
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(drop)
        ) # 16x1x62
        init.kaiming_uniform_(self.conv4c[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv4c[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv5c = nn.Sequential(
            nn.Conv2d(16, 16, (1, 4), padding=(0, 2), groups=16, bias=False), # 16x1x62
            nn.Conv2d(16, 16, (1, 1), bias=False), # 16x1x62
            nn.BatchNorm2d(16), 
            # nn.ELU(),
            nn.PReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop)
        ) # 16x1x15
        init.kaiming_uniform_(self.conv5c[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv5c[1].weight, mode='fan_in', nonlinearity='relu')

        self.conv_32_16_b = nn.Conv2d(32, 16, (1,1))
        init.kaiming_uniform_(self.conv_32_16_b.weight, mode='fan_in', nonlinearity='relu')

        self.conv_32_16_c = nn.Conv2d(32, 16, (1,1))
        init.kaiming_uniform_(self.conv_32_16_c.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv_all = nn.Conv2d(48, 16, (1,1))
        init.kaiming_uniform_(self.conv_all.weight, mode='fan_in', nonlinearity='relu')

        self.classifier = nn.Linear(16*15, 4, bias=True)
        init.kaiming_uniform_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        
        x_128 = self.conv1a(x)
        x_64 = self.conv1b(x)
        x_32 = self.conv1c(x)

        x = torch.cat((x_128, x_64, x_32), dim = 1)

        x = self.conv_24_8(x)

        x = self.conv2(x)

        x_a = self.conv3a(x)

        x_b = self.conv3b(x)
        x_c = self.conv3c(x)
        x_c = self.conv4c(x_c)

        x_b = torch.cat((x_b, x_c), dim = 1)
        x_b = self.conv_32_16_b(x_b)

        x_c = torch.cat((x_c, x_b), dim = 1)
        x_c = self.conv_32_16_c(x_c)

        x_b = self.conv4b(x_b)
        x_c = self.conv5c(x_c)

        x_con = torch.cat((x_a, x_b, x_c), dim = 1)

        x_con = self.conv_all(x_con)
        x_con = x_con.view(-1, 16*15)
        x_con = self.classifier(x_con)
        return x_con