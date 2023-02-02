from re import X
import torch
import torch.nn as nn

### Ablation Study - Point Cloud 만 실험    
class pointcloud_model(nn.Module):
    def __init__(self,device):
        super().__init__()
 
        self.pt_model = Face_Detection_Model(3,get_features=True).to(device)            
        self.fc = nn.Linear(512,1).to(device)
        
    def forward(self,cloud):
        feature = self.pt_model(cloud)
        logit = self.fc(feature)
        return logit
    
### Ablation Study - Depth 만 실험    
class depth_model(nn.Module):
    def __init__(self,device):
        super().__init__()
 
        self.depth_model = Face_Detection_Model(1,get_features=True).to(device)            
        self.fc = nn.Linear(512,1).to(device)
        
    def forward(self,depth):
        feature = self.depth_model(depth)
        logit = self.fc(feature)
        return logit
    
       
## RGB / Cloud 따로 실험 v1
class rgbp_v1_twostep_model(nn.Module):
    def __init__(self,device):
        super().__init__()

        self.rgb_model = Face_Detection_Model(3,get_features=True).to(device)    
        self.pt_model = Face_Detection_Model(3,get_features=True).to(device)            
        self.fc = nn.Linear(1024,1).to(device)
        
    def forward(self,rgb,cloud):
        rgb_feature = self.rgb_model(rgb)
        cloud_feature = self.pt_model(cloud)
        features = torch.concat([rgb_feature,cloud_feature],axis=1)
        logit = self.fc(features)
        return logit

# RGB + Cloud 같이 실험  (v2 로 실험)    
class rgbp_v2_twostep_model(nn.Module):
    def __init__(self,device):
        super().__init__()

        self.rgb_pt_model = Face_Detection_Model(6,get_features=True).to(device)    
        self.fc = nn.Linear(512,1).to(device)
        
    def forward(self,rgb,cloud):
        rgb_pt = torch.concat([rgb,cloud],axis=1)
        features = self.rgb_pt_model(rgb_pt)
        logit = self.fc(features)
        return logit


## RGB / Depth 따로 실험  v1  
class rgbd_v1_twostep_model(nn.Module):
    def __init__(self,device):
        super().__init__()
        # Model 생성 및 아키텍쳐 출력   
        self.rgb_model = Face_Detection_Model(3,get_features=True).to(device)    
        self.depth_model = Face_Detection_Model(1,get_features=True).to(device)    
        # deleteBatchnorm(self.depth_model)
        self.fc = nn.Linear(1024,1).to(device)
        
    def forward(self,rgb,depth):
        rgb_feature = self.rgb_model(rgb)
        depth_feature = self.depth_model(depth)
        features = torch.concat([rgb_feature,depth_feature],axis=1)
        logit = self.fc(features)
        return logit    
        
### RGB + Depth 로 실험 v2   
class rgbd_v2_twostep_model(nn.Module):
    def __init__(self,device):
        super().__init__() 
        self.rgb_depth_model = Face_Detection_Model(4,get_features=False).to(device)    
        
    def forward(self,rgb,depth):
        rgb_depth = torch.concat([rgb,depth],axis=1)
        logit = self.rgb_depth_model(rgb_depth)    
        return logit           
        
        
## RGB / Depth / Cloud 따로 실험 v1 
class rgbdp_v1_twostep_model(nn.Module):
    def __init__(self,device):
        super().__init__()
        # Model 생성 및 아키텍쳐 출력   
        self.rgb_model = Face_Detection_Model(3,get_features=True).to(device)    
        self.depth_model = Face_Detection_Model(1,get_features=True).to(device)    
        self.pt_model = Face_Detection_Model(3,get_features=True).to(device)   
        self.fc = nn.Linear(1536,1).to(device)
        
    def forward(self,rgb,depth,cloud):
        rgb_feature = self.rgb_model(rgb)
        depth_feature = self.depth_model(depth)
        cloud_feature = self.pt_model(cloud)
        features = torch.concat([rgb_feature,depth_feature,cloud_feature],axis=1)
        logit = self.fc(features)
        return logit

### RGB / Depth + Point Cloud v2
class rgbdp_v2_twostep_model(nn.Module):
    def __init__(self,device):
        super().__init__()
        # Model 생성 및 아키텍쳐 출력   
        self.rgb_model = Face_Detection_Model(3,get_features=True).to(device)    
        self.pt_and_depth_model = Face_Detection_Model(4,get_features=True).to(device)     
        self.fc = nn.Linear(1024,1).to(device)
        
    def forward(self,rgb,depth,cloud):
        rgb_feature = self.rgb_model(rgb)
        cloud_depth_feature = self.pt_and_depth_model(torch.concat([cloud,depth],axis=1))
        
        features = torch.concat([rgb_feature,cloud_depth_feature],axis=1)
        print(f"rgb feature: {rgb_feature.shape}")
        print(f"pt+depth feature: {cloud_depth_feature.shape}")
        print(f"fusion feature: {features.shape}")

        logit = self.fc(features)
        # return features
        return logit   


### RGB + Depth + Point Cloud 같이 실험 v3
class rgbdp_v3_twostep_model(nn.Module):
    def __init__(self,device):
        super().__init__()
        # Model 생성 및 아키텍쳐 출력   
#         # original
#         self.rgb_cloud_depth_model = Face_Detection_Model(7,get_features=True).to(device)       
#         self.fc = nn.Linear(512,1).to(device)
        
        # Revision
        self.rgb_cloud_depth_model = Face_Detection_Model(7,get_features=False).to(device)       
        
        
    def forward(self,rgb,depth,cloud):
#         # original
#         feature = self.rgb_cloud_depth_model(torch.concat([rgb,cloud,depth],axis=1))
#         logit = self.fc(feature)
#         # return feature

        print(f"early fusion feature: {torch.concat([rgb,cloud,depth],axis=1).shape}")
        logit = self.rgb_cloud_depth_model(torch.concat([rgb,cloud,depth],axis=1))



        return logit 
        
        
########################################################

### RGB + Depth / Cloud 로 실험    
# class rgbdp_twostep_model(nn.Module):
#     def __init__(self,device):
#         super().__init__()
  
#         self.rgb_depth_model = Face_Detection_Model(4,get_features=True).to(device)     
#         self.pt_model = Face_Detection_Model(3,get_features=True).to(device)    
#         self.fc = nn.Linear(1024,1).to(device)
        
#     def forward(self,rgb,depth,cloud):
        
#         rgb_depth = torch.concat([rgb,depth],axis=1)
#         rgb_depth_feature = self.rgb_depth_model(rgb_depth)    
#         cloud_feature = self.pt_model(cloud)
#         features = torch.concat([rgb_depth_feature,cloud_feature],axis=1)
#         logit = self.fc(features)
#         return logit

########################################################

        
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# ResNet34 사용 
def Face_Detection_Model(inputdata_channel=3, get_features=False):
    return ResNet(BasicBlock, [3,4,6,3], inputdata_channel=inputdata_channel, get_features=get_features)
     
    # if get_features:
    #     return feature
    # else:
    #     return logit, feature
     
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
    
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
          
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    # 이진분류 모델이므로 num_classes 를 1로 수정
    def __init__(self, block, layers, inputdata_channel, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, get_features=False):
        super(ResNet, self).__init__()
        
        self.get_features=get_features
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(inputdata_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)  
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         conv1x1(self.inplanes, planes * block.expansion, stride),
        #         norm_layer(planes * block.expansion),
        #     )
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )          

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        if self.get_features : 
            return x
        else : 
            out = self.fc(x)
            return out

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)









# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion),
#         )

#         # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
#         self.shortcut = nn.Sequential()

#         self.relu = nn.ReLU()

#         # projection mapping using 1x1conv
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )

#     def forward(self, x):
#         x = self.residual_function(x) + self.shortcut(x)
#         x = self.relu(x)
#         return x


# class BottleNeck(nn.Module):
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         self.relu = nn.ReLU()

#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels*BottleNeck.expansion)
#             )
            
#     def forward(self, x):
#         x = self.residual_function(x) + self.shortcut(x)
#         x = self.relu(x)
#         return x

# class ResNet(nn.Module):
#     # 본 모델은 이진부류이므로 num_classes 를 2로 수정 + model type 을 위한 parameter 추가 
#     #def __init__(self, block, num_block, num_classes=2, init_weights=True, inputdata_channel=3):
#     def __init__(self, block, num_block, num_classes=2, init_weights=True):
#         super().__init__()

#         self.in_channels=64

#         inputdata_channel=3
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(inputdata_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
      
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         # weights inittialization
#         if init_weights:
#             self._initialize_weights()

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self,x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         x = self.conv3_x(output)
#         x = self.conv4_x(x)
#         x = self.conv5_x(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

#     # define weight initialization function
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

# def resnet18():
#     return ResNet(BasicBlock, [2,2,2,2])

# def resnet34():
#     return ResNet(BasicBlock, [3,4,6,3])

# def resnet50():
#     return ResNet(BottleNeck, [3,4,6,3])

# def resnet101():
#     return ResNet(BottleNeck, [3,4,23,3])

# def resnet152():
#     return ResNet(BottleNeck, [3,8,36,3])

