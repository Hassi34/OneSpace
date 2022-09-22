import tensorflow as tf

def download_base_model(MODEL_ARCHITECTURE, IMAGE_SIZE):
    if MODEL_ARCHITECTURE=="VGG16":
        base_model = tf.keras.applications.vgg16.VGG16(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="MobileNetV3Large":
        base_model = tf.keras.applications.MobileNetV3Large(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="MobileNetV3Small":
        base_model = tf.keras.applications.MobileNetV3Small(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)

    elif MODEL_ARCHITECTURE=="DenseNet201":
        base_model = tf.keras.applications.densenet.DenseNet201(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="EfficientNetV2L":
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="EfficientNetV2S":
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="ResNet50":
        base_model = tf.keras.applications.resnet50.ResNet50(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="ResNet50V2":
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="ResNet152V2":
        base_model = tf.keras.applications.resnet_v2.ResNet152V2(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="VGG19":
        base_model = tf.keras.applications.vgg19.VGG19(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    elif MODEL_ARCHITECTURE=="Xception":
        base_model = tf.keras.applications.xception.Xception(
                input_shape=IMAGE_SIZE,
                weights="imagenet",
                include_top=False)
    return base_model