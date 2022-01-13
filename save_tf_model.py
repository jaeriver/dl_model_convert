import tensorflow as tf
from tensorflow.keras.applications import ( 
    xception,
    vgg16,
    vgg19,
    resnet,
    resnet50,
    resnet_v2,
    inception_v3,
    inception_resnet_v2,
    mobilenet,
    densenet,
    nasnet,
    mobilenet_v2,
    efficientnet
)
model_type = 'efficientnet_b7'
saved_model_dir = f'{model_type}'

models = {
#     'xception':xception.Xception(weights='imagenet'),
#     'vgg16':vgg16.VGG16(weights='imagenet'),
#     'vgg19':vgg19.VGG19(weights='imagenet'),
#     'resnet50':resnet50.ResNet50(weights='imagenet'),
#     'resnet101':resnet.ResNet101(weights='imagenet'),
#     'resnet152':resnet.ResNet152(weights='imagenet'),
#     'resnet50_v2':resnet_v2.ResNet50V2(weights='imagenet'),
#     'resnet101_v2':resnet_v2.ResNet101V2(weights='imagenet'),
#     'resnet152_v2':resnet_v2.ResNet152V2(weights='imagenet'),
#     'inception_v3':inception_v3.InceptionV3(weights='imagenet'),
#     'inception_resnet_v2':inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
    'mobilenet':mobilenet.MobileNet(weights='imagenet'),
#     'densenet121':densenet.DenseNet121(weights='imagenet'),
#     'densenet169':densenet.DenseNet169(weights='imagenet'),
#     'densenet201':densenet.DenseNet201(weights='imagenet'),
#     'nasnetlarge':nasnet.NASNetLarge(weights='imagenet'),
#     'nasnetmobile':nasnet.NASNetMobile(weights='imagenet'),
#     'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet'),
#     'efficientnet_b0': efficientnet.EfficientNetB0(weights='imagenet'),
#     'efficientnet_b1': efficientnet.EfficientNetB1(weights='imagenet'),
#     'efficientnet_b2': efficientnet.EfficientNetB2(weights='imagenet'),
#     'efficientnet_b3': efficientnet.EfficientNetB3(weights='imagenet'),
#     'efficientnet_b4': efficientnet.EfficientNetB4(weights='imagenet'),
#     'efficientnet_b5': efficientnet.EfficientNetB5(weights='imagenet'),
#     'efficientnet_b6': efficientnet.EfficientNetB6(weights='imagenet'),
#     'efficientnet_b7': efficientnet.EfficientNetB7(weights='imagenet')
}

model = models[model_type]
model.save(saved_model_dir)
