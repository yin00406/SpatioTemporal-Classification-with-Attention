import sys
import config
import os
import numpy as np
import torch
from torch.autograd import Function

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class UNET(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_steps):
        super(UNET,self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv4_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(128, 128, 3, padding=1) 
        self.conv5_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.conv5_2 = torch.nn.Conv2d(256, 256, 3, padding=1)

        self.unpool4 = torch.nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2)
        self.upconv4_1 = torch.nn.Conv2d(256, 128, 3, padding=1)
        self.upconv4_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.unpool3 = torch.nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(64 , 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(32 , 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)

        self.out = torch.nn.Conv2d(16, out_channels, kernel_size=1, padding=0)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def crop_and_concat(self, x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
        offset_2, offset_3 = (x1_shape[2]-x2_shape[2])//2, (x1_shape[3]-x2_shape[3])//2
        x1_crop = x1[:, :, offset_2:offset_2+x2_shape[2], offset_3:offset_3+x2_shape[3]]
        return torch.cat([x1_crop, x2], dim=1)

    def forward(self, x, time_step):
        x = x[:,time_step,:,:,:]

        conv1 = self.relu(self.conv1_2(self.relu(self.conv1_1(x))))
        maxpool1 = self.maxpool(conv1)
        conv2 = self.relu(self.conv2_2(self.relu(self.conv2_1(maxpool1))))
        maxpool2 = self.maxpool(conv2)
        conv3 = self.relu(self.conv3_2(self.relu(self.conv3_1(maxpool2))))
        maxpool3 = self.maxpool(conv3)
        conv4 = self.relu(self.conv4_2(self.relu(self.conv4_1(maxpool3))))
        maxpool4 = self.maxpool(conv4)
        conv5 = self.relu(self.conv5_2(self.relu(self.conv5_1(maxpool4))))

        unpool4 = self.unpool4(conv5)
        upconv4 = self.relu(self.upconv4_2(self.relu(self.upconv4_1(self.crop_and_concat(conv4, unpool4)))))
        unpool3 = self.unpool3(upconv4)
        upconv3 = self.relu(self.upconv3_2(self.relu(self.upconv3_1(self.crop_and_concat(conv3, unpool3)))))
        unpool2 = self.unpool2(upconv3)
        upconv2 = self.relu(self.upconv2_2(self.relu(self.upconv2_1(self.crop_and_concat(conv2, unpool2)))))
        unpool1 = self.unpool1(upconv2)
        upconv1 = self.relu(self.upconv1_2(self.relu(self.upconv1_1(self.crop_and_concat(conv1, unpool1)))))

        out = self.out(upconv1)
        return out

class STC_BN_TACA(UNET):
    def __init__(self, in_channels, out_channels, time_steps):
        super(STC_BN_TACA,self).__init__(in_channels, out_channels, time_steps)

        self.lstm = torch.nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.attention = torch.nn.Linear(512, 1)
        self.unpool4 = torch.nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.attention_ca = torch.nn.Linear(time_steps * 64 * 64, 1)
        self.time_steps = time_steps

        # Define BatchNorm2d layers
        self.bn1_1 = torch.nn.BatchNorm2d(16)
        self.bn1_2 = torch.nn.BatchNorm2d(16)
        self.bn2_1 = torch.nn.BatchNorm2d(32)
        self.bn2_2 = torch.nn.BatchNorm2d(32)
        self.bn3_1 = torch.nn.BatchNorm2d(64)
        self.bn3_2 = torch.nn.BatchNorm2d(64)
        self.bn4_1 = torch.nn.BatchNorm2d(128)
        self.bn4_2 = torch.nn.BatchNorm2d(128)
        self.bn5_1 = torch.nn.BatchNorm2d(256)
        self.bn5_2 = torch.nn.BatchNorm2d(256)

        # BatchNorm for the LSTM output
        self.bn_lstm = torch.nn.BatchNorm1d(512)

        # BatchNorm2d layers for the upconv layers
        self.bn_upconv1_1 = torch.nn.BatchNorm2d(16)
        self.bn_upconv1_2 = torch.nn.BatchNorm2d(16)
        self.bn_upconv2_1 = torch.nn.BatchNorm2d(32)
        self.bn_upconv2_2 = torch.nn.BatchNorm2d(32)
        self.bn_upconv3_1 = torch.nn.BatchNorm2d(64)
        self.bn_upconv3_2 = torch.nn.BatchNorm2d(64)
        self.bn_upconv4_1 = torch.nn.BatchNorm2d(128)
        self.bn_upconv4_2 = torch.nn.BatchNorm2d(128)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        batch_size, seq_len, channels, input_patch_size, input_patch_size = x.shape

        x = x.permute(0, 2, 1, 3, 4)
        x_reshape = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        x_reshape = x_reshape.reshape(x_reshape.shape[0] * x_reshape.shape[1], x_reshape.shape[2])
        tanh = torch.tanh(x_reshape)
        attn = self.attention_ca(tanh).view(-1, channels, 1, 1)
        attn_ca = torch.nn.functional.softmax(torch.squeeze(torch.nn.functional.avg_pool2d(attn, 1)), dim=1)
        x = (attn_ca.view(-1, 1) * x_reshape).view(-1, channels,
                                                   self.time_steps * 64 * 64).view(
            -1, channels, self.time_steps, 64, 64).permute(0, 2, 1, 3, 4)

        x = x.reshape(-1, channels, input_patch_size, input_patch_size)

        conv1 = self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(x))))))
        maxpool1 = self.maxpool(conv1)
        maxpool1 = self.dropout(maxpool1)

        conv2 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(maxpool1))))))
        maxpool2 = self.maxpool(conv2)
        maxpool2 = self.dropout(maxpool2)

        conv3 = self.relu(self.bn3_2(self.conv3_2(self.relu(self.bn3_1(self.conv3_1(maxpool2))))))
        maxpool3 = self.maxpool(conv3)
        maxpool3 = self.dropout(maxpool3)

        conv4 = self.relu(self.bn4_2(self.conv4_2(self.relu(self.bn4_1(self.conv4_1(maxpool3))))))
        maxpool4 = self.maxpool(conv4)
        maxpool4 = self.dropout(maxpool4)

        conv5 = self.relu(self.bn5_2(self.conv5_2(self.relu(self.bn5_1(self.conv5_1(maxpool4))))))
        conv5 = self.dropout(conv5)

        shape_enc = conv5.shape
        conv5 = conv5.view(-1, self.time_steps, conv5.shape[1], conv5.shape[2]*conv5.shape[3])
        conv5 = conv5.permute(0, 3, 1, 2)
        conv5 = conv5.reshape(conv5.shape[0]*conv5.shape[1], self.time_steps, 256)
        lstm, _ = self.lstm(conv5)
        lstm = self.relu(self.bn_lstm(lstm.reshape(-1, 512)))
        attention_weights = torch.nn.functional.softmax(torch.squeeze(torch.nn.functional.avg_pool2d(self.attention(torch.tanh(lstm)).view(-1, shape_enc[2], shape_enc[3], self.time_steps).permute(0, 3, 1, 2), shape_enc[2])), dim=1)
        context = torch.sum((attention_weights.view(-1, 1, 1, self.time_steps).repeat(1, shape_enc[2], shape_enc[3], 1).view(-1, 1) * lstm).view(-1, self.time_steps, 512), dim=1).view(-1, shape_enc[2], shape_enc[3], 512).permute(0, 3, 1, 2)
        attention_weights_fixed = attention_weights.detach()
        context = self.dropout(context)

        unpool4 = self.unpool4(context)
        agg_conv4 = torch.sum(attention_weights_fixed.view(-1, self.time_steps, 1, 1, 1) * conv4.view(-1, self.time_steps, conv4.shape[1], conv4.shape[2], conv4.shape[3]), dim=1)
        upconv4 = self.relu(self.bn_upconv4_2(self.upconv4_2(self.relu(self.bn_upconv4_1(self.upconv4_1(self.crop_and_concat(agg_conv4, unpool4)))))))
        upconv4 = self.dropout(upconv4)

        unpool3 = self.unpool3(upconv4)
        agg_conv3 = torch.sum(attention_weights_fixed.view(-1, self.time_steps, 1, 1, 1) * conv3.view(-1, self.time_steps, conv3.shape[1], conv3.shape[2], conv3.shape[3]), dim=1)
        upconv3 = self.relu(self.bn_upconv3_2(self.upconv3_2(self.relu(self.bn_upconv3_1(self.upconv3_1(self.crop_and_concat(agg_conv3, unpool3)))))))
        upconv3 = self.dropout(upconv3)

        unpool2 = self.unpool2(upconv3)
        agg_conv2 = torch.sum(attention_weights_fixed.view(-1, self.time_steps, 1, 1, 1) * conv2.view(-1, self.time_steps, conv2.shape[1], conv2.shape[2], conv2.shape[3]), dim=1)
        upconv2 = self.relu(self.bn_upconv2_2(self.upconv2_2(self.relu(self.bn_upconv2_1(self.upconv2_1(self.crop_and_concat(agg_conv2, unpool2)))))))
        upconv2 = self.dropout(upconv2)

        unpool1 = self.unpool1(upconv2)
        agg_conv1 = torch.sum(attention_weights_fixed.view(-1, self.time_steps, 1, 1, 1) * conv1.view(-1, self.time_steps, conv1.shape[1], conv1.shape[2], conv1.shape[3]), dim=1)
        upconv1 = self.relu(self.bn_upconv1_2(self.upconv1_2(self.relu(self.bn_upconv1_1(self.upconv1_1(self.crop_and_concat(agg_conv1, unpool1)))))))
        upconv1 = self.dropout(upconv1)

        out = self.out(upconv1)

        return out
