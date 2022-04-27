# learning to compose objects by parts with Pythorch3D
# using a multi-step model to add parts to a mesh so its rendering matches an input image

import torch
import torch.nn as nn
import torch.nn.functional as F



# # decoder RNN / GRU: 
# # from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.fci = nn.Linear(self.output_size, self.hidden_size)
#         self.fco = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input, hidden):
#         input = self.fci(input)
#         # gru input (seq, batch_size, emb_dim)
#         output, hidden = self.gru(input, hidden)
#         output = self.fco(output)
#         return output, hidden


# This model takes as input the rendered silhouette image frame 128x128x1
# and outputs: position (3: x,y,z), size (3: sx,sy,sz)
# note: we ignore part number and color for now, so 6 outputs!
class meshNetPartsV1(nn.Module):

    def __init__(self, outSize=6):
        super(meshNetPartsV1, self).__init__()
        self.outSize = outSize
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*13*13, 128)
        self.fc2 = nn.Linear(128, self.outSize)

    def forward(self, x):
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x


# This model takes as input the rendered silhouette image frame 128x128x1
# and outputs: position (3: x,y,z), size (3: sx,sy,sz)
# note: we ignore part number and color for now, so 6 outputs!
# with RNN decoder
class meshNetPartsV2(nn.Module):

    def __init__(self, outSize=6):
        super(meshNetPartsV2, self).__init__()
        self.outSize = outSize
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*13*13, 128)
        # self.fc2 = nn.Linear(128, self.outSize)
        # decoder
        self.fco1 = nn.Linear(2*128, 128)
        self.fco2 = nn.Linear(128, 6)
        # self.decoder = DecoderRNN(hidden_size=128, output_size=6)

    def forward(self, im, idx):
        h = F.max_pool2d(F.relu(self.conv1(im)), 2)
        # print(h.shape)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        # print(h.shape)
        h = F.max_pool2d(F.relu(self.conv3(h)), 2)
        # print(h.shape)
        h = torch.flatten(h, 1) # flatten all dimensions except the batch dimension
        # print(h.shape)
        h = F.relu(self.fc1(h)) # embedded image is h to RNN!
        # print(h.shape)
        # x, _ = self.decoder(idx, h.unsqueeze(0))
        c = torch.cat([h,idx], dim=1)
        x = F.relu(self.fco1(c))
        x = self.fco2(x)

        return x



# This model takes as input the rendered silhouette image frame 128x128x1
# and outputs ALL: position (3: x,y,z), scale (3: sx,sy,sz), rotation (rx,ry,rz)
# note: we ignore part number and color for now, so num_parts*9 outputs!
class meshNetPartsV3(nn.Module):

    def __init__(self, num_parts, tex=True):
        super(meshNetPartsV3, self).__init__()
        self.num_parts = num_parts
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*13*13, 1024)
        # self.fo1 = nn.Linear(256, 256)
        self.fo2_pos = nn.Linear(1024, self.num_parts*3) # position classifier
        self.fo2_scale = nn.Linear(1024, self.num_parts*3) # scale classifier
        self.fo2_rot = nn.Linear(1024, self.num_parts*3) # rotation classifier
        self.fo2_type = nn.Linear(1024, self.num_parts*4) # type of part classifier (4 parts 1-hot)
         # texture classifiers:
        self.tc1 = nn.Linear(1024, 1024)
        self.tc2 = nn.Linear(1024, 98*self.num_parts*3)
        self.tex = tex

    def forward(self, x):
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fo1(x))
        # print(x.shape)
        position = self.fo2_pos(x)
        scale = F.relu(self.fo2_scale(x)) # this needs to be >0
        rotation = self.fo2_rot(x)
        ntype = F.softmax(self.fo2_type(x), dim=1) # needs softmax to be diffentiable
        
        # textures:
        if self.tex:
            t = torch.relu(self.tc1(x))
            t = torch.sigmoid(self.tc2(t)) # has to be between [0,1]
            return position, scale, rotation, ntype, t # t = textures
        else:
            return position, scale, rotation, ntype


# This model takes as input the rendered RGB image frame 128x128x3
# and outputs: [part number (1),] position (3: x,y,z), size (3: sx,sy,sz), [color (3: RGB)]