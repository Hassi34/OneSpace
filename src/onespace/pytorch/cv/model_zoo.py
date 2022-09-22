from torchvision import models

def download_base_model(MODEL_ARCHITECTURE):
    if MODEL_ARCHITECTURE=="AlexNet":
        base_model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="ConvnNeXt":
        base_model = models.convnext_large(weights = models.ConvNeXt_Large_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="DenseNet121":
        base_model = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="DenseNet201":
        base_model = models.densenet201(weights = models.DenseNet201_Weights.DEFAULT)    

    elif MODEL_ARCHITECTURE=="EfficientNet_b7":
        base_model = models.efficientnet_b7(weights = models.EfficientNet_B7_Weights.DEFAULT)  

    elif MODEL_ARCHITECTURE=="EfficientNet_v2_s":
        base_model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT)  

    elif MODEL_ARCHITECTURE=="EfficientNet_v2_m":
        base_model = models.efficientnet_v2_m(weights = models.EfficientNet_V2_M_Weights.DEFAULT)  

    elif MODEL_ARCHITECTURE=="EfficientNet_v2_l":
        base_model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights.DEFAULT)  

    elif MODEL_ARCHITECTURE=="GoogleNet":
        base_model = models.googlenet(weights = models.GoogLeNet_Weights.DEFAULT)  

    elif MODEL_ARCHITECTURE=="Inception_v3":
        base_model = models.inception_v3(weights = models.Inception_V3_Weights.DEFAULT)  
        base_model.aux_logits=False

    elif MODEL_ARCHITECTURE=="MnasNet0_5":
        base_model = models.mnasnet0_5(weights = models.MNASNet0_5_Weights.DEFAULT)  

    elif MODEL_ARCHITECTURE=="MnasNet1_3":
        base_model = models.mnasnet1_3(weights = models.MNASNet1_3_Weights.DEFAULT)  

    elif MODEL_ARCHITECTURE=="MobileNet_v2":
        base_model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="MobileNet_v3_large":
        base_model = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="MobileNet_v3_small":
        base_model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="RegNet_y_32gf":
        base_model = models.regnet_y_32gf(weights = models.RegNet_Y_32GF_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="ResNet18":
        base_model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="ResNet34":
        base_model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="ResNet50":
        base_model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="ResNet152":
        base_model = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="ResNext101_32x8d":
        base_model = models.resnext101_32x8d(weights = models.ResNeXt101_32X8D_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="ShuffleNet_v2_x1_5":
        base_model = models.shufflenet_v2_x1_5(weights = models.ShuffleNet_V2_X1_5_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="SqueezeNet1_0":
        base_model = models.squeezenet1_0(weights = models.SqueezeNet1_0_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="SwinTransformer":
        base_model = models.swin_s(weights = models.Swin_S_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="SqueezeNet1_0":
        base_model = models.squeezenet1_0(weights = models.SqueezeNet1_0_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="VGG11":
        base_model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="VGG13":
        base_model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="VGG16":
        base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="VGG19":
        base_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="VisionTransformer":
        base_model = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)

    elif MODEL_ARCHITECTURE=="Wide_Resnet50_2":
        base_model = models.wide_resnet50_2(weights = models.Wide_ResNet50_2_Weights.DEFAULT)
    return base_model