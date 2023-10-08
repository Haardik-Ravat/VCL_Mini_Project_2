# imports
import torch.nn as nn
import torch.nn.functional as F

class Rnn3D(nn.Module):

    #define the learnable paramters by calling the respective modules (nn.Conv2d, nn.MaxPool2d etc.)
    def __init__(self):
        super(Rnn3D, self).__init__()

        #calling conv3d module for convolution
        conv1 = nn.Conv3d(in_channels = 16, out_channels = 50, kernel_size = 2, stride = 1)

        #calling MaxPool3d module for max pooling with downsampling of 2
        pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)

        conv2 =  nn.Conv3d(in_channels = 50, out_channels = 100, kernel_size = (1, 3, 3), stride = 1)

        pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2)

        self.feat_extractor=nn.Sequential(conv1,nn.ReLU(),pool1,conv2,nn.ReLU(),pool2)

        self.rnn = nn.LSTM(input_size=5625, hidden_size=128, num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(128, 5)



    def forward(self, x):

        b_z, ts, c, h, w = x.shape

        y = self.feat_extractor(x)

        # reinstating the batchsize and frames
        y = y.view(b_z,ts,-1)
        #output has a size of 8x16x128 - basically we have the output for each frame of each clip.
        outp, (hn, cn) = self.rnn(y)
        # We only need the RNN/LSTM output of the last frame since it incorporates all the frame knowledge
        out = self.fc1(outp[:,-1,:])
        return out
