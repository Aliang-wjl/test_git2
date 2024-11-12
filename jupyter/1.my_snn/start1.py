# 改python文件包含了许多训练时需要导入的包，绘图函数，训练函数，测试函数，ser_resnet网络，spiking resnet网络

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# # InteractiveShell.ast_node_interactivity = "last_expr"

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.datasets.asl_dvs import ASLDVS

from spikingjelly.datasets import split_to_train_test_set as sptt
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer,monitor

from torch import nn,amp,optim # 自动微分
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR,LambdaLR,CosineAnnealingWarmRestarts,CosineAnnealingLR

# from tqdm import tqdm  # 这个进度条用于pycharm
from tqdm.notebook import tqdm  # tqdm 进度条显示,这个更好看，用于notebook
from PIL import Image

import matplotlib.pyplot as plt
import gc,math,torch,os,sys,time,datetime,tonic,psutil,argparse,random
import numpy as np


# # del data
# gc.collect()
# process = psutil.Process(os.getpid())
# print(process.memory_info().rss)  # 查看当前进程的内存使用
# torch.cuda.empty_cache()

_seed_ = 2024
random.seed(2024)
np.random.seed(_seed_)
# use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.manual_seed(_seed_)  
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def plot_data(train_data,test_data,name):
    # Plot Loss
    fig = plt.figure(facecolor="w", figsize=(5, 5))
    plt.plot(train_data)
    plt.plot(test_data)
    plt.title(name+" Curves")
    plt.legend(["Train", "Test"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

# 学习率调度函数
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子， 这个倍率因子乘以学习率等于我们最终使用的lr
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return LambdaLR(optimizer, lr_lambda=f)

# 权重分配函数
def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

#     print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def trains(num_epochs, net, train_loader,test_loader, optimizer, device, loss_function, writer, T_train = None, max_test_acc = 0, 
          scaler = None, encoder = encoding.PoissonEncoder(), T = 0, num_classes = 10, data_type = 0, pt_dir = "",
          lr_scheduler = None):
    '''
    data_type: 数据维度是否具有时间维度，0表示数据本身没有时间维度，要么在SNN中增加时间维度，要么在训练时增加时间维度
           1表示有时间维度，此时数据维度为[N,T,.....]，而在SNN中，需要的数据维度为[T,N,.....]，所以需要转换
    T != 0 ,代表需要encoder了，且是在网络外进行的时间步循环, 是数据本身没有时间步，且网络也没有增加时间步这一维度
    '''
    train_accs = []
    train_losss = []
    test_accs = []
    test_losss = []
    loss_MSE = F.mse_loss
    
    for epoch in range(num_epochs):
        start_time = time.time()
        _ = net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        datas = tqdm(iter(train_loader),file=sys.stdout)
        for img, label in datas:
            # 梯度清零
            optimizer.zero_grad()
            img = img.to(device)
            # 选择几个时间步用于训练
            if T_train:
                sec_list = np.random.choice(img.shape[1], T_train, replace=False)
                sec_list.sort()
                img = img[:, sec_list]
            
            if data_type != 0:
                img = img.transpose(0, 1)  # 将数据维度由  [N, T, C, H, W] 变为  [T, N, C, H, W]
            
            label = label.to(device)
            label_onehot = F.one_hot(label, num_classes).float().to(device)  # 将标签变为独热编码

            # 混合精度训练
            if scaler is not None:
                with amp.autocast(device_type="cuda"):
#                 with amp.autocast():
                    out_fr = 0.
                    if T != 0:
                        for _ in range(T):
                            encoded_img = encoder(data)
                            out_fr += net(encoded_img) # 将每次输出的脉冲进行相加
                        out_fr = out_fr / T # 求平均值
                    else:
#                         print(img.type())
                        out_fr = net(img)
                    if loss_function == loss_MSE:
                        loss = F.mse_loss(out_fr, label_onehot)
                    else:
                        loss = loss_function(out_fr, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                if T != 0:
                    for _ in range(T):
                        encoded_img = encoder(data)
                        out_fr += net(encoded_img) # 将每次输出的脉冲进行相加
                    out_fr = out_fr / T # 求平均值
                else:
                    out_fr = net(img)
#                 print(out_fr.shape,label_onehot.shape)
                if loss_function == loss_MSE:
                    loss = F.mse_loss(out_fr, label_onehot)
                else:
                    loss = loss_function(out_fr, label)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            # 正确率的计算方法如下。  认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的。
            functional.reset_net(net)
            
            
#             current_lr = optimizer.param_groups[0]['lr']
            if lr_scheduler != None:
                lr_scheduler.step() # 学习率的优化
#             print(f"current_lr = {current_lr}")
           
        
        end_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"epoch = {epoch}, current_lr = {current_lr}")
        print(f"train_single_time = {end_time - start_time:.4f}")
        print(f"loss = {train_loss/train_samples:.4f}")
        print(f"acc = {train_acc/train_samples*100:.2f}%")
        train_accs.append(train_acc/train_samples)
        train_losss.append(train_loss/train_samples)
        # 写入数据
        writer.add_scalar('train_loss', train_losss[-1], epoch)
        writer.add_scalar('train_acc', train_accs[-1], epoch)

        test_acc,test_loss = evaluates("test", net, test_loader,device,loss_function, encoder = encoding.PoissonEncoder(),
                                       T = T, num_classes = num_classes, data_type = data_type)
        test_accs.append(test_acc)
        test_losss.append(test_loss)
        # 写入数据
        writer.add_scalar('test_loss', test_losss[-1], epoch)
        writer.add_scalar('test_acc', test_accs[-1], epoch)
        
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True
        
        # 检查点
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc 
            }
        
        # 保存最好的模型
        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))
            print("2 save_max model param")
        
        # 保存最近的一次模型
        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))
        print("1 save_latest model param")
        
        # 使用脚本执行时输出的Python文件
        for item in sys.argv:
            print(item, end=' ')    
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (num_epochs - epoch - 1))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        print("\n") 
    return train_accs,train_losss,test_accs,test_losss


# 测试函数
def evaluates(name, model, data_loader, device,loss_function, encoder = encoding.PoissonEncoder(), T = 0, num_classes = 10, data_type = 0):
    """
    T = 0，代表具有网络的输出已经在时间维度上进行了平均 ------ T != 0，表示网络没有时间维度这个概念，这种情况下一般需要先编码，最后平均。
    data_type代表的是数据类型，0代表正常数据集，1代表具有时间维度的数据集。
    这个测试函数仅适用于使用独热编码的mse损失函数。
    """
    loss_MSE = F.mse_loss
    acc = 0
    losss = 0
    data_num = 0
    _ = model.eval()

    datas = tqdm(iter(data_loader),file=sys.stdout)
    with torch.no_grad():
        for data, label in datas:
            data = data.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, num_classes).float()  # 将标签变为独热编码
            if data_type != 0:
                data = data.transpose(0, 1)
            
            out_fr = 0.
            if T == 0:
                out_fr = model(data)
            else:
                for _ in range(T):
                    encoded_img = encoder(data)
                    out_fr += model(encoded_img) # 将每次输出的脉冲进行相加
                out_fr = out_fr / T # 求平均值
            
            if loss_function == loss_MSE:
                loss = F.mse_loss(out_fr, label_onehot)
            else:
                loss = loss_function(out_fr, label)
            data_num += label.numel()
            losss += loss.item() * label.numel()
            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            acc += (out_fr.argmax(1) == label).float().sum().item()
            
            # 每次重置网络
            functional.reset_net(model)
    
    print(f"{name} All Acc: {acc/data_num * 100:.2f}%")
    print(f"{name} All Loss: {losss/data_num:.2f}")
    return acc/data_num,losss/data_num


# 用来测试DVSGesture数据集的一个基础网络
class DVSGestureNet(nn.Module):
    def __init__(self, inp_channels = 3, num_classes = 11, channels=128):
        super().__init__()
        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = inp_channels
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(neuron.IFNode(surrogate_function=surrogate.ATan()))
#             neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())   两种神经元的替换使用
            conv.append(layer.MaxPool2d(2, 2))
        

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.VotingLayer(int(110 / num_classes))
        )
        # 设置网络的模式，是单步还是多步，一般都是多步
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor):
        output = self.conv_fc(x)
        return output.mean(dim=0)
    
    
# SEW_ResNet原始网络及面对不同数据集所进行的网络变换以及数据操作的变换
class BasicBlock(nn.Module):
    """
    实现ResNet网络的第一个基础架构
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(BasicBlock, self).__init__()
        
        self.connect_f = connect_f
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.layer1 = nn.Sequential(
            layer.Conv2d(inplanes, planes, kernel_size=3, stride = stride, padding=1, bias=False),
            norm_layer(planes),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer2 = nn.Sequential(
            layer.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            norm_layer(planes),
            neuron.IFNode(surrogate_function=surrogate.ATan()))

        self.downsample = downsample
        
        # 设置所有的层为多步模式
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
#         print(out.shape)
#         print(identity.shape)
        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class Bottleneck(nn.Module):
    """
    实现ResNet网络的瓶颈架构
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(Bottleneck, self).__init__()
        
        self.connect_f = connect_f
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        width = int(planes * (base_width / 64.)) * groups
        
        self.layer1 = nn.Sequential(
            layer.Conv2d(inplanes, width, kernel_size=1, bias=False),
            norm_layer(width),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer2 = nn.Sequential(
            layer.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, groups = groups, dilation = dilation, bias=False),
            norm_layer(width),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer3 = nn.Sequential(
            layer.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False),
            norm_layer(planes * self.expansion),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
   
        self.downsample = downsample
        
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

# 对这两个架构的最后一层中的 BN 层进行设置，权重为0，且当连接方式为AND时，偏差置为1
def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.layer3.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.layer3.module[1].bias, 1)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.layer2.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.layer2.module[1].bias, 1)
        else:
            pass

# 设计SEWResNet网络，原始的网络
class SEWResNet(nn.Module):

    def __init__(self, block, layers, input_channels = 3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f=None, drops = [0,0,0,0], p = [0.5,0.5,0.5,0.5], data_type = 0, net_type = 0):
        """
        data_type：输入的数据是没有时间维度的，还是具有时间维度的
        net_type：使用sew_resnet还是sew_resnet2
        """
        super(SEWResNet, self).__init__()
        
        # 是否使用扩张率代替步长，一般这个操作应该用不上
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.T = T
        self.connect_f = connect_f
        self.groups = groups
        self.base_width = width_per_group
        self.drops = drops
        self.data_type = data_type
        self.net_type = net_type
        
        self.layer01 = nn.Sequential(
            layer.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes))
        
        self.layer01s = nn.Sequential(
            layer.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes))
        functional.set_step_mode(self.layer01s, step_mode='m')
        
        self.layer02 = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2, padding=1))
        functional.set_step_mode(self.layer02, step_mode='m')

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.dp1 = layer.Dropout(p[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.dp2 = layer.Dropout(p[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.dp3 = layer.Dropout(p[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.dp4 = layer.Dropout(p[3])
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        # 最后一层的通道输出需要乘以扩张率
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 使用扩张率代替步长
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # 步长不等于1的时候，当然我们需要使用下采样层，当步长为1，但是在一个Block中，输入通道数不等于输出通道数的时候，
        # 我们仍然需要使用下采样层，因为我们每一个block块的输出一定是 planes * block.expansion，输入一定是self.inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    layer.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride = stride, bias=False),
                    norm_layer(planes * block.expansion),
                    neuron.IFNode(surrogate_function=surrogate.ATan()))
            functional.set_step_mode(downsample, step_mode='m')
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        # 注意经过一个block之后，输入会发生通道的变化。
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if self.data_type == 0:
            if self.net_type == 0:
                x = self.layer01(x)
                x.unsqueeze_(0) # 这一步经过实验可以去掉，因为x.repeat会自动添加维度
                x = x.repeat(self.T, 1, 1, 1, 1)  # T,B,C,H,W
            else:
                x.unsqueeze_(0) # 这一步经过实验可以去掉，因为x.repeat会自动添加维度
                x = x.repeat(self.T, 1, 1, 1, 1)  # T,B,C,H,W
                x = self.layer01s(x)
        else:
            x = self.layer01s(x)
        x = self.layer02(x)
        x = self.layer1(x)
        if self.drops[0]:
            x = self.dp1(x)
        x = self.layer2(x)
        if self.drops[1]:
            x = self.dp2(x)
        x = self.layer3(x)
        if self.drops[2]:
            x = self.dp3(x)
        x = self.layer4(x)
        if self.drops[3]:
            x = self.dp4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2) # 将C,H,W合并为一体
        return self.fc(x.mean(dim=0)) # 首先在T这个维度求平均，接着使用Linear层

    def forward(self, x):
        return self._forward_impl(x)
    
def _sew_resnet(block, layers, **kwargs):
    model = SEWResNet(block, layers, **kwargs)
    return model


def sew_resnet18(**kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152(**kwargs):
    return _sew_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)



# spiking_resnet网络的实现
class SP_BasicBlock(nn.Module):
    """
    实现ResNet网络的第一个基础架构
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(SP_BasicBlock, self).__init__()
        
        self.connect_f = connect_f
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.layer1 = nn.Sequential(
            layer.Conv2d(inplanes, planes, kernel_size=3, stride = stride, padding=1, bias=False),
            norm_layer(planes),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer2 = nn.Sequential(
            layer.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            norm_layer(planes),
            )
        
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())

        self.downsample = downsample
        
        # 设置所有的层为多步模式
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        out = self.layer2(self.layer1(x))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        return self.sn2(out)


class SP_Bottleneck(nn.Module):
    """
    实现ResNet网络的瓶颈架构
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(SP_Bottleneck, self).__init__()
        
        self.connect_f = connect_f
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        width = int(planes * (base_width / 64.)) * groups
        
        self.layer1 = nn.Sequential(
            layer.Conv2d(inplanes, width, kernel_size=1, bias=False),
            norm_layer(width),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer2 = nn.Sequential(
            layer.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, groups = groups, dilation = dilation, bias=False),
            norm_layer(width),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer3 = nn.Sequential(
            layer.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False),
            norm_layer(planes * self.expansion))
        
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan())
   
        self.downsample = downsample
        
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        out = self.layer3(self.layer2(self.layer1(x)))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity

        return self.sn3(out)

# 对这两个架构的最后一层中的 BN 层进行设置，权重为0，且当连接方式为AND时，偏差置为1
def SP_zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.layer3.module[1].weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.layer2.module[1].weight, 0)

class SpikingResNet(nn.Module):

    def __init__(self, block, layers, input_channels = 3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f=None):
        super(SpikingResNet, self).__init__()
        
        # 是否使用扩张率代替步长，一般这个操作应该用不上
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.T = T
        self.connect_f = connect_f
        self.groups = groups
        self.base_width = width_per_group
        
        self.layer01 = nn.Sequential(
            layer.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes))
        
        self.layer02 = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2, padding=1))
        functional.set_step_mode(self.layer02, step_mode='m')

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        # 最后一层的通道输出需要乘以扩张率
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            SP_zero_init_blocks(self, connect_f)
        
        

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 使用扩张率代替步长
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # 步长不等于1的时候，当然我们需要使用下采样层，当步长为1，但是在一个Block中，输入通道数不等于输出通道数的时候，
        # 我们仍然需要使用下采样层，因为我们每一个block块的输出一定是 planes * block.expansion，输入一定是self.inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    layer.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride = stride, bias=False),
                    norm_layer(planes * block.expansion))
            functional.set_step_mode(downsample, step_mode='m')
#             functional.set_step_mode(downsample, step_mode='m')   
# 经过测试，如果需要整个网络为多步，只需要init函数尾部设置functional.set_step_mode(self, step_mode='m')
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        # 注意经过一个block之后，输入会发生通道的变化。
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)
    


    def _forward_impl(self, x):
        x = self.layer01(x)
        # 将前面的层当成数据的变化，但是一般来说应该是先输入脉冲之后，在进行各种变化，所以从直觉上这种输入是否符合逻辑？
        x.unsqueeze_(0) # 这一步经过实验可以去掉，因为x.repeat会自动添加维度
        x = x.repeat(self.T, 1, 1, 1, 1)  # T,B,C,H,W
        x = self.layer02(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2) # 将C,H,W合并为一体
        return self.fc(x.mean(dim=0)) # 首先在T这个维度求平均，接着使用Linear层

    def forward(self, x):
        return self._forward_impl(x)
    
def _spiking_resnet(block, layers, **kwargs):
    model = SpikingResNet(block, layers, **kwargs)
    return model


def spiking_resnet18(**kwargs):
    return _spiking_resnet(SP_BasicBlock, [2, 2, 2, 2], **kwargs)


def spiking_resnet34(**kwargs):
    return _spiking_resnet(SP_BasicBlock, [3, 4, 6, 3], **kwargs)


def spiking_resnet50(**kwargs):
    return _spiking_resnet(SP_Bottleneck, [3, 4, 6, 3], **kwargs)


def spiking_resnet101(**kwargs):
    return _spiking_resnet(SP_Bottleneck, [3, 4, 23, 3], **kwargs)


def spiking_resnet152(**kwargs):
    return _spiking_resnet(SP_Bottleneck, [3, 8, 36, 3], **kwargs)