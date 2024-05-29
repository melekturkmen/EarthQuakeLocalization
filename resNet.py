

import torch
import torch.nn.functional as F

#build a block that can be re-used through out the network
#the block contains a skip connection = downsample that is an optional parameter
#note that in the forward pass, skip connnection is applied 
#directly to the inpur of that layer (a^l) and takes it to two layers ahead 

#Referenced from the paper of Complex CNN (Ritea, 2018)
class ResNet(torch.nn.Module):
    def __init__(self, Height, Width,**kwargs): #freq or time
        super().__init__() #downsample as constructer parameter
        
        self.freq = kwargs.get("freq_flag")
        self.stat_info = kwargs.get("add_stat_info")
        self.output_size = kwargs.get("gt_select")
        
        if self.freq:
            #conv1
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,7), stride=1, padding = (1,3)) #W=51, H=9
            self.batchnorm1 = torch.nn.BatchNorm2d(num_features = 32)  
            height, width = self._calculate_output_shape((3,7), (1,1),(1,3),(int(Height), int(Width)))
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride = (1,2)) #W=25,H=17
            height, width = self._calculate_output_shape(kernel_size=(3,3), stride=(1,2), padding= (0,0), input_shape=(height, width))
            #conv2
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,7), stride=1, padding = (1,3)) #W=25, H=9
            height, width = self._calculate_output_shape( kernel_size=(3,7), stride=(1,1), padding = (1,3), input_shape=(height, width))
            self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 32) 
            self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(3,1), stride = (2,1))  #W=25,H=8
            height, width = self._calculate_output_shape( kernel_size=(3,1), stride=(2,1), padding = (0,0), input_shape=(height, width))
            #residual 3
            self.conv1x1r3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=1)
            self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding = (1,1)) #W=25,H=8
            self.batchnorm3 = torch.nn.BatchNorm2d(num_features = 64)
            height, width = self._calculate_output_shape(kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            #conv4
            self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding = (1,1)) #W=25,H=8
            height, width = self._calculate_output_shape( kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            self.batchnorm4 = torch.nn.BatchNorm2d(num_features = 64)
            self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(1,3), stride= (1,2)) #W=12,H=8
            height, width = self._calculate_output_shape( kernel_size=(1,3), stride=(1,2), padding = (0,0), input_shape=(height, width))
            
            #residual 4
            self.conv1x1r4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=1)
            self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding = (1,1)) #W=12,H=8
            height, width = self._calculate_output_shape( kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            self.batchnorm5 = torch.nn.BatchNorm2d(num_features = 128)
            #conv6
            self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding = (1,1)) #W=12,H=8
            self.batchnorm6 = torch.nn.BatchNorm2d(num_features = 128)
            height, width = self._calculate_output_shape( kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            
            #residual 5
            self.conv1x1r5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=1)
            self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding = (1,1)) #W=12,H=8
            self.batchnorm7 = torch.nn.BatchNorm2d(num_features = 256)
            height, width = self._calculate_output_shape(kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            #conv8
            self.conv8 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding = (1,1)) #W=12,H=8
            
            self.batchnorm8 = torch.nn.BatchNorm2d(num_features = 256) #W=12,H=8
            height, width = self._calculate_output_shape( kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            #fc + Concatenation block    
            self.linear1 = torch.nn.LazyLinear(kwargs.get("FC_size"))
            self.linear2 = torch.nn.LazyLinear(len(self.output_size))
            
            self.dropout1 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            self.dropout2 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            self.dropout3 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))

        else:
            #ENCODER
            #conv1
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7,1), stride=1,padding=(3,0)) 
            self.batchnorm1 = torch.nn.BatchNorm2d(num_features = 32)  
             
            #conv2
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(7,1), stride=1,padding=(3,0)) 
            self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 32) 
            self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(4,1), stride = (4,1)) 
            
            
            #end of plain network
           
            #residual 3
            self.conv1x1r3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=1) 
            self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), stride=1,padding=(1,0)) 
            self.batchnorm3 = torch.nn.BatchNorm2d(num_features = 64)
           
            #conv4
            self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), stride=1,padding=(1,0))
            self.batchnorm4 = torch.nn.BatchNorm2d(num_features = 64)
            self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2,1), stride= (2,1)) 
           
           
            #residual 4
            self.conv1x1r4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=1)
            self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,1), stride=1,padding=(1,0)) 
            self.batchnorm5 = torch.nn.BatchNorm2d(num_features = 128)
            #conv6
            self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,1), stride=1,padding=(1,0)) 
            self.batchnorm6 = torch.nn.BatchNorm2d(num_features = 128)
            self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(2,1), stride= (2,1)) 
           
            #residual 5
            self.conv1x1r5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=1)
            self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,1), stride=1,padding=(1,0)) 
            self.batchnorm7 = torch.nn.BatchNorm2d(num_features = 256)
            #conv8
            self.conv8 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,1), stride=1,padding=(1,0))
            self.batchnorm8 = torch.nn.BatchNorm2d(num_features = 256)
            self.maxpool5 = torch.nn.MaxPool2d(kernel_size=(2,1), stride= (2,1))
            
            self.linear1 = torch.nn.LazyLinear(kwargs.get("FC_size"))
            self.linear2 = torch.nn.LazyLinear(len(self.output_size))
            
            self.dropout1 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            self.dropout2 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            self.dropout3 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            

                        
            
    def _calculate_output_shape(self, kernel_size, stride, padding, input_shape):
            # Formula for calculating output shape of a conv or pool layer
            height = ((input_shape[0] + 2*padding[0] - kernel_size[0]) / stride[0]) + 1
            width = ((input_shape[1] + 2*padding[1] - kernel_size[1]) / stride[1]) + 1
            return int(height), int(width)
        
    def forward(self, x, data,**kwargs): 
        self.freq = kwargs.get("freq_flag")
        self.stat_info = kwargs.get("add_stat_info")
        if self.freq:
            x = x.to(torch.float32)
            x = self.conv1(x)
            x = self.maxpool1(x) #torch.Size([batchSize, 256, 17, 25])
            x = self.batchnorm1(x)
            x = F.relu(x)
            
            x = self.conv2(x) #torch.Size([batchSize, 256, 17, 25])
            x = self.maxpool2(x)#torch.Size([batchSize, 256, 8, 25])
            x = self.batchnorm2(x)
            x = F.relu(x)
            
            #residual 3
            #conv+relu+conv+relu+maxpool
            x = self.conv1x1r3(x)
            residual = x
            x = self.conv3(x) #torch.Size([batchSize, 256, 8, 25])
            x = self.batchnorm3(x)
            x = F.relu(x)
            
            x = self.conv4(x) #torch.Size([batchSize, 256, 8, 25])
            x = self.batchnorm4(x)
            x += residual            
            x = F.relu(x)

            x = self.maxpool3(x) #torch.Size([batchSize, 256, 8, 12])
            
            #residual 4
            #conv+relu+conv+relu+maxpool
            x = self.conv1x1r4(x)
            residual = x
            x = self.conv5(x) #torch.Size([batchSize, 256, 8, 12])
            x = self.batchnorm5(x)
            x = F.relu(x)
            
            x = self.conv6(x) #torch.Size([batchSize, 256, 8, 12])
            x = self.batchnorm6(x)
            x += residual            
            x = F.relu(x)
            # x = self.dropout2(x)
            #residual 5
            #conv+relu+conv+relu+maxpool
            x = self.conv1x1r5(x)
            residual = x
            x = self.conv7(x) #torch.Size([batchSize, 256, 8, 12])
            x = self.batchnorm7(x)
            x = F.relu(x)
            
            x = self.conv8(x) #torch.Size([batchSize, 256, 8, 12])
            x = self.batchnorm8(x)
            x += residual
            # x = self.dropout3(x)
            
            x = x.flatten(start_dim=1)
            
            if self.stat_info:
            # Concatanate two streams 1024 + 3 (stat_info)
                x = torch.cat((x, data), dim=1)
            
            x = self.dropout1(x)
            x = self.linear1(x)
            x = self.linear2(x)
            return x
        else:
           x = x.to(torch.float32)
           x = self.conv1(x)           
           x = self.batchnorm1(x)
           x = F.relu(x)
           
           x = self.conv2(x)
           x = self.maxpool2(x)
           x = self.batchnorm2(x)
           x = F.relu(x)
           
           #residual 3
           #conv+relu+conv+relu+maxpool
           x = self.conv1x1r3(x)
           residual = x
           x = self.conv3(x)
           x = self.batchnorm3(x)
           x = F.relu(x)
           
           x = self.conv4(x) 
           x = self.batchnorm4(x)
           x += residual
           x = F.relu(x)
           # x = self.dropout1(x)
           x = self.maxpool3(x)
           
           #residual 4
           #conv+relu+conv+relu+maxpool
           x = self.conv1x1r4(x)
           residual = x
           x = self.conv5(x)
           x = self.batchnorm5(x)
           x = F.relu(x)
           
           x = self.conv6(x)
           x = self.batchnorm6(x)
           x += residual
           x = F.relu(x)
           # x = self.dropout2(x)
           x = self.maxpool4(x)
           
           #residual 5
           #conv+relu+conv+relu+maxpool
           x = self.conv1x1r5(x)
           residual = x
           x = self.conv7(x)
           x = self.batchnorm7(x)
           x = F.relu(x)
           
           x = self.conv8(x)           
           x = self.batchnorm8(x)
           x += residual
           # x = self.dropout3(x)
           x = self.maxpool5(x)
           
           x = x.flatten(start_dim=1)  
           
           if self.stat_info:
           # Concatanate two streams 1024 + 3 (stat_info)
               x = torch.cat((x, data), dim=1)
           
           x = self.linear1(x)
           x = self.linear2(x)

           return x
       
    
    def transfer_pretrained(self, pretrained):
        if self.freq:
            self.conv1 = pretrained.conv1
            self.batchnorm1 = pretrained.batchnorm1
            self.conv2 = pretrained.conv2
            self.batchnorm2 = pretrained.batchnorm2
            #res1
            self.conv3 = pretrained.conv3
            self.batchnorm3 = pretrained.batchnorm3
            self.conv4 = pretrained.conv4
            self.batchnorm4 = pretrained.batchnorm4
            #res2
            self.conv5 = pretrained.conv5
            self.batchnorm5 = pretrained.batchnorm5
            
            self.conv6 = pretrained.conv6
            self.batchnorm6 = pretrained.batchnorm6
            #res4
            self.conv7 = pretrained.conv7
            self.batchnorm7 = pretrained.batchnorm7
            
            self.conv8 = pretrained.conv8
            self.batchnorm8 = pretrained.batchnorm8
            
            self.linear1 = pretrained.linear1
        else:
            self.conv1 = pretrained.conv1
            self.batchnorm1 = pretrained.batchnorm1
            self.conv2 = pretrained.conv2
            self.batchnorm2 = pretrained.batchnorm2
            #res1
            self.conv3 = pretrained.conv3
            self.batchnorm3 = pretrained.batchnorm3
            self.conv4 = pretrained.conv4
            self.batchnorm4 = pretrained.batchnorm4
            #res2
            self.conv5 = pretrained.conv5
            self.batchnorm5 = pretrained.batchnorm5
            
            self.conv6 = pretrained.conv6
            self.batchnorm6 = pretrained.batchnorm6
            #res4
            self.conv7 = pretrained.conv7
            self.batchnorm7 = pretrained.batchnorm7
            
            self.conv8 = pretrained.conv8
            self.batchnorm8 = pretrained.batchnorm8
            
            self.linear1 = pretrained.linear1
            
class EncoderResNet(torch.nn.Module):
    def __init__(self, Height, Width,**kwargs): #freq or time
        super().__init__() #downsample as constructer parameter
        
        self.freq = kwargs.get("freq_flag")
        self.stat_info = kwargs.get("add_stat_info")
        self.output_size = kwargs.get("gt_select")
        
        if self.freq:
            #conv1
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,7), stride=1, padding = (1,3)) #W=51, H=9
            self.batchnorm1 = torch.nn.BatchNorm2d(num_features = 32,track_running_stats=False)  
            height, width = self._calculate_output_shape((3,7), (1,1),(1,3),(int(Height), int(Width)))
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(3,3), stride = (1,2)) #W=25,H=17
            height, width = self._calculate_output_shape(kernel_size=(3,3), stride=(1,2), padding= (0,0), input_shape=(height, width))
            #conv2
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,7), stride=1, padding = (1,3)) #W=25, H=9
            height, width = self._calculate_output_shape( kernel_size=(3,7), stride=(1,1), padding = (1,3), input_shape=(height, width))
            self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 32,track_running_stats=False) 
            self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(3,1), stride = (2,1))  #W=25,H=8
            height, width = self._calculate_output_shape( kernel_size=(3,1), stride=(2,1), padding = (0,0), input_shape=(height, width))
            #residual 3
            self.conv1x1r3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=1)
            self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding = (1,1)) #W=25,H=8
            self.batchnorm3 = torch.nn.BatchNorm2d(num_features = 64,track_running_stats=False)
            height, width = self._calculate_output_shape(kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            #conv4
            self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding = (1,1)) #W=25,H=8
            height, width = self._calculate_output_shape( kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            self.batchnorm4 = torch.nn.BatchNorm2d(num_features = 64,track_running_stats=False)
            self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(1,3), stride= (1,2)) #W=12,H=8
            height, width = self._calculate_output_shape( kernel_size=(1,3), stride=(1,2), padding = (0,0), input_shape=(height, width))
            
            #residual 4
            self.conv1x1r4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=1)
            self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding = (1,1)) #W=12,H=8
            height, width = self._calculate_output_shape( kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            self.batchnorm5 = torch.nn.BatchNorm2d(num_features = 128,track_running_stats=False)
            #conv6
            self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding = (1,1)) #W=12,H=8
            self.batchnorm6 = torch.nn.BatchNorm2d(num_features = 128,track_running_stats=False)
            height, width = self._calculate_output_shape( kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            
            #residual 5
            self.conv1x1r5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=1)
            self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding = (1,1)) #W=12,H=8
            self.batchnorm7 = torch.nn.BatchNorm2d(num_features = 256,track_running_stats=False)
            height, width = self._calculate_output_shape(kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))
            #conv8
            self.conv8 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding = (1,1)) #W=12,H=8
            
            self.batchnorm8 = torch.nn.BatchNorm2d(num_features = 256,track_running_stats=False) #W=12,H=8
            height, width = self._calculate_output_shape( kernel_size=(3,3), stride=(1,1), padding = (1,1), input_shape=(height, width))

            
            self.dropout1 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            self.dropout2 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            self.dropout3 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))

        else:
            #ENCODER
            #conv1
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7,1), stride=1,padding=(3,0)) 
            self.batchnorm1 = torch.nn.BatchNorm2d(num_features = 32,track_running_stats=False)  
             
            #conv2
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(7,1), stride=1,padding=(3,0)) 
            self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 32,track_running_stats=False) 
            self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(4,1), stride = (4,1)) 
            
            
            #end of plain network
           
            #residual 3
            self.conv1x1r3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=1) 
            self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), stride=1,padding=(1,0)) 
            self.batchnorm3 = torch.nn.BatchNorm2d(num_features = 64,track_running_stats=False)
           
            #conv4
            self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), stride=1,padding=(1,0))
            self.batchnorm4 = torch.nn.BatchNorm2d(num_features = 64,track_running_stats=False)
            self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2,1), stride= (2,1)) 
           
           
            #residual 4
            self.conv1x1r4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=1)
            self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,1), stride=1,padding=(1,0)) 
            self.batchnorm5 = torch.nn.BatchNorm2d(num_features = 128,track_running_stats=False)
            #conv6
            self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,1), stride=1,padding=(1,0)) 
            self.batchnorm6 = torch.nn.BatchNorm2d(num_features = 128,track_running_stats=False)
            self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(2,1), stride= (2,1)) 
           
            #residual 5
            self.conv1x1r5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=1)
            self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,1), stride=1,padding=(1,0)) 
            self.batchnorm7 = torch.nn.BatchNorm2d(num_features = 256,track_running_stats=False)
            #conv8
            self.conv8 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,1), stride=1,padding=(1,0))
            self.batchnorm8 = torch.nn.BatchNorm2d(num_features = 256,track_running_stats=False)
            self.maxpool5 = torch.nn.MaxPool2d(kernel_size=(2,1), stride= (2,1))
            
            self.dropout1 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            self.dropout2 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))
            self.dropout3 = torch.nn.Dropout(p = kwargs.get("dropout_rate"))   
            
    def forward(self, x, data,**kwargs): 
        self.freq = kwargs.get("freq_flag")
        self.stat_info = kwargs.get("add_stat_info")
        if self.freq:
            x = x.to(torch.float32)
            x = self.conv1(x)
            x = self.maxpool1(x) #torch.Size([batchSize, 256, 17, 25])
            x = self.batchnorm1(x)
            x = F.relu(x)
            
            x = self.conv2(x) #torch.Size([batchSize, 256, 17, 25])
            x = self.maxpool2(x)#torch.Size([batchSize, 256, 8, 25])
            x = self.batchnorm2(x)
            x = F.relu(x)
            
            #residual 3
            #conv+relu+conv+relu+maxpool
            x = self.conv1x1r3(x)
            residual = x
            x = self.conv3(x) #torch.Size([batchSize, 256, 8, 25])
            x = self.batchnorm3(x)
            x = F.relu(x)
            
            x = self.conv4(x) #torch.Size([batchSize, 256, 8, 25])
            x = self.batchnorm4(x)
            x += residual            
            x = F.relu(x)
            x = self.dropout1(x)
            x = self.maxpool3(x) #torch.Size([batchSize, 256, 8, 12])
            
            #residual 4
            #conv+relu+conv+relu+maxpool
            x = self.conv1x1r4(x)
            residual = x
            x = self.conv5(x) #torch.Size([batchSize, 256, 8, 12])
            x = self.batchnorm5(x)
            x = F.relu(x)
            
            x = self.conv6(x) #torch.Size([batchSize, 256, 8, 12])
            x = self.batchnorm6(x)
            x += residual            
            x = F.relu(x)
            x = self.dropout2(x)
            #residual 5
            #conv+relu+conv+relu+maxpool
            x = self.conv1x1r5(x)
            residual = x
            x = self.conv7(x) #torch.Size([batchSize, 256, 8, 12])
            x = self.batchnorm7(x)
            x = F.relu(x)
            
            x = self.conv8(x) #torch.Size([batchSize, 256, 8, 12])
            x = self.batchnorm8(x)
            x += residual
            x = self.dropout3(x)
            
           
            return x
        else:
           x = x.to(torch.float32)
           x = self.conv1(x)           
           x = self.batchnorm1(x)
           x = F.relu(x)
           
           x = self.conv2(x)
           x = self.maxpool2(x)
           x = self.batchnorm2(x)
           x = F.relu(x)
           
           #residual 3
           #conv+relu+conv+relu+maxpool
           x = self.conv1x1r3(x)
           residual = x
           x = self.conv3(x)
           x = self.batchnorm3(x)
           x = F.relu(x)
           
           x = self.conv4(x) 
           x = self.batchnorm4(x)
           x += residual
           x = F.relu(x)
           x = self.dropout1(x)
           x = self.maxpool3(x)
           
           #residual 4
           #conv+relu+conv+relu+maxpool
           x = self.conv1x1r4(x)
           residual = x
           x = self.conv5(x)
           x = self.batchnorm5(x)
           x = F.relu(x)
           
           x = self.conv6(x)
           x = self.batchnorm6(x)
           x += residual
           x = F.relu(x)
           x = self.dropout2(x)
           x = self.maxpool4(x)
           
           #residual 5
           #conv+relu+conv+relu+maxpool
           x = self.conv1x1r5(x)
           residual = x
           x = self.conv7(x)
           x = self.batchnorm7(x)
           x = F.relu(x)
           
           x = self.conv8(x)           
           x = self.batchnorm8(x)
           x += residual
           x = self.dropout3(x)
           x = self.maxpool5(x)

           return x        
       
class TransferModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(TransferModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x,stat_info,**kwargs):
        with torch.no_grad():
            encoder_output = self.encoder(x,stat_info,**kwargs)
        x = self.decoder(encoder_output,stat_info,**kwargs)
        return x  
     
class DecoderResNet(torch.nn.Module):
    def __init__(self, Height, Width,**kwargs): #freq or time
        super().__init__() #downsample as constructer parameter

        self.stat_info = kwargs.get("add_stat_info")
        self.output_size = kwargs.get("gt_select")

        self.linear1 = torch.nn.LazyLinear(kwargs.get("FC_size"))
        self.linear2 = torch.nn.LazyLinear(len(self.output_size))
    def forward(self, x, data,**kwargs): 
        
        self.stat_info = kwargs.get("add_stat_info")
        
        x = x.flatten(start_dim=1)
             
        if self.stat_info:
             # Concatanate two streams 1024 + 3 (stat_info)
             x = torch.cat((x, data), dim=1)
             
             
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
class TimeAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(TimeAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = torch.nn.Linear(input_size, hidden_size)
        self.key = torch.nn.Linear(input_size, hidden_size)
        self.value = torch.nn.Linear(input_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.out = torch.nn.Linear(hidden_size, input_size)

    def forward(self, x):

        b, c, t = x.size()
        x = x.permute(0, 2, 1)
        x = x.reshape(b*t, c)


        query = self.query(x).view(b, t, self.num_heads, self.head_size)
        key = self.key(x).view(b, t, self.num_heads, self.head_size)
        value = self.value(x).view(b, t, self.num_heads, self.head_size)

        query = query.permute(0, 2, 1, 3).contiguous().view(b*self.num_heads, t, self.head_size)
        key = key.permute(0, 2, 1, 3).contiguous().view(b*self.num_heads, t, self.head_size)
        value = value.permute(0, 2, 1, 3).contiguous().view(b*self.num_heads, t, self.head_size)

        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / self.head_size**0.5
        attn = torch.nn.Softmax(dim=2)(attn)
        attn = self.dropout(attn)

        out = torch.bmm(attn, value)
        out = out.view(b, self.num_heads, t, self.head_size)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, t, self.num_heads*self.head_size)


        out = self.out(out)

        return out
