import torch
import torch.nn as nn

architecture_config = [
    # 1. Bölüm: (kernel_size, out_channels, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    
    # 2. Bölüm:
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    
    # 3. Bölüm: 4 Kez Tekrar Eden Bloklar [(conv1), (conv2), repeat_count]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    
    # 4. Bölüm: 2 Kez Tekrar Eden Bloklar + Son Conv Katmanları
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        return self.conv_block(x)

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, S=7, B=2, C=20):
        super().__init__()
        self.in_channels = in_channels
        self.S = S
        self.B = B
        self.C = C
        self.darknet = self._create_conv_layers(architecture_config)
        self.fcs = self._create_fcs()

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3]
                    )
                )
                in_channels = x[1]

            elif isinstance(x, str):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers.append(
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    )
                    layers.append(
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    )
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))
        )

    def forward(self, x):
        x = self.darknet(x)
        x = self.fcs(x)
        return x.reshape(-1, self.S, self.S, self.C + self.B * 5)

if __name__ == "__main__":
    model = YOLOv1(in_channels=3, S=7, B=2, C=20)
    x = torch.randn((2, 3, 448, 448))
    out = model(x)
    print("Model Girdi Boyutu :", x.shape)
    print("Model Çıktı Boyutu:", out.shape)