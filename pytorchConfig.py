#--------------------------------------------------------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------------------------------------------------------
data_dir = "data_folder"
train_folder_name = "train"
val_folder_name = "val"
transfer_learning = True
model_architecture = "MobileNet_v3_small" # available_selections = ["AlexNet", "ConvnNeXt", "DenseNet121", "DenseNet201", "EfficientNet_b7", "EfficientNet_v2_s", "EfficientNet_v2_m", "EfficientNet_v2_l", "Wide_Resnet50_2",
                                         #                          "GoogleNet", "Inception_v3", "MnasNet0_5", "MnasNet1_3", "MobileNet_v2", "MobileNet_v3_large", "MobileNet_v3_small", "RegNet_y_32gf", "ResNet18",
                                         #                         "ResNet34", "ResNet50", "ResNet152", "ResNext101_32x8d", "ShuffleNet_v2_x1_5", "SqueezeNet1_0", "VGG11", "VGG13", "VGG16", "VGG19", "VisionTransformer"]
                                         
augmentation = False
epochs = 1
batch_size = 32
input_shape = (224, 224, 3)
lr_scheduler = "ExponentialLR"           # available_selections = ["OneCycleLR","StepLR", "LambdaLR", "ExponentialLR"]
optimizer = "RMSprop"                        # available_selections = ["SGD", "Adam", "Adadelta", "Adagrad", "RMSprop"]
validation_split = 0.20
grad_clip = 1                        # Should be a value ranging from 0.5 to 1.0 OR None
weight_decay = 0.001                 # Should be a value ranging from 0.5 to 1.0

#--------------------------------------------------------------------------------------------------------------------------------
# Artifacts (Directory names to store the results & resources, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
project_name = 'cvpytorch1'
artifacts_dir = "Artifacts"
model_dir = "Models"
plots_dir = "Plots"
model_name = "my_test_model"
experiment_name = "Test exp with def"
plot_name = "results_plot"
model_ckpt_dir = "Model Checkpoints"
callbacked_model_name = "model_ckpt"

#--------------------------------------------------------------------------------------------------------------------------------
# Logs (Directory names to record logs, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
logs_dir = "Logs"
tensorboard_root_log_dir = "Tensorboard Logs"
csv_logs_dir = "CSV Logs"
csv_logs_file = "cv_test_logs.csv"
comments = "this is comments"
executed_by = 'Hasnain'

#--------------------------------------------------------------------------------------------------------------------------------
# Database Integration
#--------------------------------------------------------------------------------------------------------------------------------
# Please Note that before making any change in this section, create a .env file and store the mongo db connection string or MySQL credentials in the environment variables 
# Guideline for creating .env is available on project description main page
from dotenv import load_dotenv
load_dotenv()
db_integration_mysql = True
db_integration_mongodb = True 

