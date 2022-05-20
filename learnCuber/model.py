# learning to compose objects by parts with Pythorch3D
# using a multi-step model to add parts to a mesh so its rendering matches an input image

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# transformer new model for cubes:
class meshTRP(nn.Module):

    def __init__(self, num_parts, nviews, nlayers=6, d_model=512, nhead=8, d_hid=512, dropout=0.5):
        super(meshTRP, self).__init__()
        self.num_parts = num_parts
        self.d_model = d_model
        # conv image encoder:
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*13*13, 512)
        # transformer encoder of all images at once:
        # encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # decoder classifiers:
        self.decoder_pos = nn.Linear(nviews*d_model, num_parts*3)
        self.decoder_size = nn.Linear(nviews*d_model, num_parts*3)


    def forward(self, x):
        # print(x.shape)
        # embed all images:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # print(x.shape)
        image_embs = F.relu(self.fc1(x))
        # print(image_embs.shape)

        # src = image_embs.unsqueeze(0) * math.sqrt(self.d_model)
        # print(src.shape)
        # src = self.pos_encoder(src)
        # output = self.transformer_encoder(src)#, src_mask)
        output = torch.flatten(image_embs, 0) # flatten all dimensions except the batch dimension
        # print(output.shape)
        pos = self.decoder_pos(output)
        size = self.decoder_size(output)

        return pos, size
