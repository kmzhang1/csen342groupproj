from .resnetv2 import ResNet18
from .wrn import wrn_40_2
from .mobilenetv2 import mobile_half
from .ShuffleNetv2 import ShuffleV2
from .GoogleNet import GoogleNetModel
from .ResNeXt import ResNeXtModel

model_dict = {
    'ResNet18': ResNet18,
    'wrn_40_2': wrn_40_2,
    'MobileNetV2': mobile_half,
    'ShuffleV2': ShuffleV2,
    'GoogleNet': GoogleNetModel,
    'ResNeXt': ResNeXtModel
}
