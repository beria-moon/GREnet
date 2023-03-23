import torch

from torch import nn
from deit_main.models import deit_tiny_distilled_patch16_224,deit_tiny_distilled


class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        #self.activation =nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        #self.dp =nn.Dropout2d(p=0.3)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        #x =self.dp(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output= self.activation(x)
        #output =self.dp(x)

        return output
class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y
    
    

class TransUNet(nn.Module):
    def __init__(self,imgsize,n_channels=3, n_classes=1):
        super(TransUNet, self).__init__()
        self.imgsize = imgsize
        self.pachsize= imgsize//16
        transformer = deit_tiny_distilled(self.imgsize)
        #transformer =  deit_tiny_distilled_patch16_224(pretrained=False)
        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1536)
        self.conv2d = nn.Conv2d(in_channels=1536, out_channels=1024, kernel_size=1, padding=0)
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.pool00_10=nn.Conv2d(filters[0], filters[0],kernel_size=3,stride=2,padding=1)
        self.pool10_20=nn.Conv2d(filters[1], filters[1],kernel_size=3,stride=2,padding=1)
        self.pool20_30=nn.Conv2d(filters[2], filters[2],kernel_size=3,stride=2,padding=1)
        self.pool30_40=nn.Conv2d(filters[3], filters[3],kernel_size=3,stride=2,padding=1)
        self.Up40_31 =nn.ConvTranspose2d(filters[4] , filters[4], kernel_size=2, stride=2)
        self.Up31_22=nn.ConvTranspose2d(filters[3] , filters[3], kernel_size=2, stride=2)
        self.Up22_13=nn.ConvTranspose2d(filters[2] , filters[2], kernel_size=2, stride=2)
        self.Up13_04=nn.ConvTranspose2d(filters[1] , filters[1], kernel_size=2, stride=2)
        self.conv0_0 = conv_block_nested(n_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        k = h//16
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool00_10(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool10_20(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool20_30(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))
        
        

        x4_0 = self.conv4_0(self.pool30_40(x3_0))#feature cnn
        emb = self.patch_embed(x)
        for i in range(12):
            emb = self.transformers[i](emb)
        feature_tf = emb.permute(0, 2, 1)
       
        feature_tf = feature_tf.view(b, 192, k, k)
        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((x4_0, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)
   

        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up40_31(feature_out)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up31_22(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up22_13(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up13_04(x1_3)], 1))
        output_1=self.final(x0_1)
        output_2=self.final(x0_2)
        output_3=self.final(x0_3)
        output_4 = self.final(x0_4)
       
        return  output_1,output_2,output_3,output_4
    
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
    
        x = self.conv(x)
        return x

        


    