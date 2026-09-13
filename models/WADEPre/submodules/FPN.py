import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(
        self,
        num_blocks,
        block=Bottleneck,
        in_channels=3,
        feature_channels=256,
        layer_channels=None,
        input_sizes=None,
    ):
        """
        多尺度输入的FPN

        Args:
            block: 残差块类型 (如 Bottleneck)
            num_blocks: 每层的残差块数量列表
            in_channels: 输入通道数 (所有尺寸的输入通道数相同)
            feature_channels: FPN特征图的统一通道数
            layer_channels: 每层处理后的通道数列表，如果为None则使用默认值[64, 128, 256, 512]
            input_sizes: 每层输入的空间尺寸列表，如[(23,23), (38,38), (68,68)]
                        注意：应该从小到大排列（从深层到浅层）
        """
        super(FPN, self).__init__()

        if layer_channels is None:
            layer_channels = [64, 128, 256, 512]

        self.num_layers = len(num_blocks)
        self.feature_channels = feature_channels
        self.input_sizes = input_sizes

        # 为每个输入尺寸创建独立的输入适配层
        self.input_adapters = nn.ModuleList()
        self.in_planes_list = []

        for i, (channels, num_block) in enumerate(zip(layer_channels, num_blocks)):
            # 输入适配：将输入通道数转换为目标通道数
            adapter = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * block.expansion),
                nn.ReLU(inplace=True),
            )
            self.input_adapters.append(adapter)

        # Bottom-up处理层 - 为每个尺寸创建独立的处理层
        self.bottom_up_layers = nn.ModuleList()
        for i, (channels, num_block) in enumerate(zip(layer_channels, num_blocks)):
            self.in_planes = channels * block.expansion
            # 所有层stride=1，因为输入已经是不同尺寸了
            layer = self._make_layer(block, channels, num_block, stride=1)
            self.bottom_up_layers.append(layer)

        # Top layer - 处理最后一层（最深层，最小尺寸）
        last_channels = layer_channels[-1] * block.expansion
        self.toplayer = nn.Conv2d(
            last_channels, feature_channels, kernel_size=1, stride=1, padding=0
        )

        # Lateral layers - 动态创建（除了最后一层）
        self.lateral_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            in_ch = layer_channels[self.num_layers - 2 - i] * block.expansion
            lateral = nn.Conv2d(
                in_ch, feature_channels, kernel_size=1, stride=1, padding=0
            )
            self.lateral_layers.append(lateral)

        # Smooth layers - 动态创建
        self.smooth_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            smooth = nn.Conv2d(
                feature_channels, feature_channels, kernel_size=3, stride=1, padding=1
            )
            self.smooth_layers.append(smooth)

        # 初始化权重
        self.init_weight()

    def init_weight(self):
        """
        FPN 初始化：使用 Kaiming 初始化（适合 ReLU）
        BatchNorm 使用标准初始化（weight=1, bias=0）
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 初始化，适合 ReLU
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def forward(self, inputs):
        """
        Args:
            inputs: list或tuple，包含多个不同尺寸的输入特征图
                   从小到大排列（从深层到浅层）
                   例如: [B,C,23,23], [B,C,38,38], [B,C,68,68]

        Returns:
            tuple: 包含所有层的输出特征图，顺序与输入相同
                  例如: ([B,F,23,23], [B,F,38,38], [B,F,68,68])
                  其中F是feature_channels
        """
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("输入必须是list或tuple，包含多个不同尺寸的特征图")

        if len(inputs) != self.num_layers:
            raise ValueError(
                f"输入层数 {len(inputs)} 与模型层数 {self.num_layers} 不匹配"
            )

        # Bottom-up - 对每个输入分别进行处理
        bottom_up_features = []
        for i, (inp, adapter, layer) in enumerate(
            zip(inputs, self.input_adapters, self.bottom_up_layers)
        ):
            # 输入适配
            feat = adapter(inp)
            # 通过残差层处理
            feat = layer(feat)
            bottom_up_features.append(feat)

        # Top-down - 从最后一层（最小尺寸）开始
        # 最顶层（最深层，最小尺寸）
        top_down_features = []
        p = self.toplayer(bottom_up_features[-1])
        top_down_features.append(p)

        # 逐层向上融合
        for i in range(self.num_layers - 1):
            # 获取对应的lateral layer和bottom-up特征
            lateral_idx = i
            bottom_up_idx = self.num_layers - 2 - i

            # 上采样并融合
            p = self._upsample_add(
                p, self.lateral_layers[lateral_idx](bottom_up_features[bottom_up_idx])
            )
            top_down_features.append(p)

        # 反转列表，使其从小到大排列（与输入顺序一致）
        top_down_features = top_down_features[::-1]

        # Smooth - 对除了最深层以外的所有层进行平滑
        output_features = []
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                # 应用smooth层
                smoothed = self.smooth_layers[i](top_down_features[i])
                output_features.append(smoothed)
            else:
                # 最深层（最小尺寸）不需要smooth
                output_features.append(top_down_features[i])
        
        output_features = [f.contiguous() for f in output_features]
        # 返回所有层的输出，顺序与输入相同（从小到大尺寸）
        return tuple(output_features)
