from torch import nn
from collections import OrderedDict
import torch

# Bottleneck 클래스는 ResNet과 같은 네트워크에서 사용되는 기본 블록
# 1x1, 3x3, 1x1의 세 개의 합성곱 층을 통해 입력을 처리
class Bottleneck(nn.Module):
    expansion = 4 # 출력 채널 수 확장을 위한 계수, 일반적으로 4로 설정

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # output size = (input size - kernel size + 2 x padding size)/stride +1

        # 첫 번째 1x1 합성곱 레이어
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # 배치 정규화
        self.relu1 = nn.ReLU(inplace=True) # 활성화 함수

        # 두 번째 3x3 합성곱 레이어
        # (64-3+2x1) / 1 + 1
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # stride가 1보다 클 경우에 평균 풀링을 적용하여 크기를 줄임
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 세 번째 1x1 합성곱 레이어
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        # 다운샘플링이 필요한 경우, 이를 위한 추가 레이어 구성
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # 입력과 출력의 차원을 맞추기 위한 다운샘플링
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x  # 스킵 연결을 위한 입력 저장

        # 첫 번째, 두 번째, 세 번째 합성곱 연산을 차례로 실행
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)  # 평균 풀링 적용
        out = self.bn3(self.conv3(out))

        # 다운샘플링이 필요하면, identity 값을 다운샘플링
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 스킵 연결로 identity를 더함
        out = self.relu3(out)  # 최종 활성화 함수 적용
        return out