import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!= -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BachNorm2d")!=-1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.activation = nn.ReLU(inplace=True)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

# same Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # , padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(60, 64, kernel_size=9, stride=1)
        self.in1_e = nn.BatchNorm2d(64, affine=True)

        self.conv2 = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.in2_e = nn.BatchNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.BatchNorm2d(128, affine=True)

        self.conv4 = ConvLayer(128, 128, kernel_size=3, stride=2)
        self.in4_e = nn.BatchNorm2d(128, affine=True)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv4 = UpsampleConvLayer(128, 128, kernel_size=3, stride=1, upsample=2)
        self.in4_d = nn.BatchNorm2d(128, affine=True)

        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.BatchNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 64, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.BatchNorm2d(64, affine=True)

        self.deconv1 = UpsampleConvLayer(64, 60, kernel_size=9, stride=1)
        self.in1_d = nn.BatchNorm2d(60, affine=True)

        self.Attn0 = Self_Attn(128)
        self.Attn1 = Self_Attn(64)
        self.Attn2 = Self_Attn(64)



    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.Attn1(y)
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = self.Attn2(y)
        y = self.tanh(self.in1_d(self.deconv1(y)))

        return y




def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!= -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BachNorm2d")!=-1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(60, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )
        # self.layer3 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ELU(inplace=True),
        #     nn.Dropout(0.25),
        # )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(4 * 64, 64),
            # nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
            nn.Softmax()
            )

    def forward(self, img):
        x1 = self.layer1(img)
        x2 = self.layer2(x1)
        xb = self.layer3(x2)
        # print(x.shape)
        x3 = xb.view(-1, 4*64)
        x4 = self.layer4(x3)
        return x4, xb, x2, x1


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layer = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layer.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layer)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, input_shape):
        super(GeneratorUNet, self).__init__()

        channels, height, width = input_shape

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        # self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.attn1 = Self_Attn(512)
        self.attn2 = Self_Attn(256)
        self.attn3 = Self_Attn(128)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        # d6 = self.down6(d5)

        # u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d4)
        # u3 = self.up3(u2, d3)
        # u4 = self.up4(u3, d2)
        # u5 = self.up5(u4, d1)

        u1 = self.up2(d5, d4)
        u2 = self.up3(u1, d3)
        u3 = self.up4(u2, d2)
        u4 = self.up5(u3, d1)


        return self.final(u4)


