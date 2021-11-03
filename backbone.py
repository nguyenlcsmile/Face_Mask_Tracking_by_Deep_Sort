from resnext.model_res import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, DenseNet169, DenseNet201

backbones = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
}

def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)