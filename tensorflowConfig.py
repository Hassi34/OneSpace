#--------------------------------------------------------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------------------------------------------------------
data_dir = "data_folder" # main directory for data
train_folder_name = "train" # name of the training folder. Pass "None" if there is no training and validation folder available
val_folder_name = "val"     # name of the validation folder pass "None" if there is no training and validation folder available
transfer_learning = False    # Pass "False" if you want to train the model from scratch
model_architecture = "MobileNetV3Small" # available_selections = ["VGG16", "MobileNetV3Large", "MobileNetV3Small", "DenseNet201", "EfficientNetV2L", "EfficientNetV2S", "ResNet50", "ResNet50V2", "ResNet152V2", "VGG19", "Xception"]
freeze_all = True           # Will freeze the weights for all the layers except for last one (Dense Layer)
freeze_till = None          # selection = [-1,-2,-3,-4,-5] => if you want to freeze weights until a specific layer, select any value from the list and observe the trainable prams
augmentation = False         # Pass True if the augmentation should be applied on the images. It helps with regularization.
epochs = 1                 # Number of Epochs
batch_size = 64
input_shape = (224, 224, 3) 
activation = "relu"         # available_selections = ["relu","selu", "swish", "elu", "gelu", "tanh"] => This param is used when there is no transferlearning involved and model is being trained from scratch 
activation_output = "softmax" # available_selections = ["softmax","sigmoid", "hard_sigmoid"]
loss_function = "binary_crossentropy" # available_selection = ["binary_crossentropy","categorical_crossentropy" , "sparse_categorical_crossentropy"]
metrics = ["AUC","Recall" ]   # available_selection = ["accuracy", "AUC", "Precision", "Recall" ]
lr_scheduler = 'InverseTimeDecay' #availabel_selection = ["InverseTimeDecay","CosineDecay", "ExponentialDecay", "CosineDecayRestarts", "PolynomialDecay"] # can be observed on tensorboard
optimizer = "Ftrl"          # available_selection = ["SGD", "Adam", "RMSprop", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
validation_split = 0.20     # The proportion of data should be used to validation. This param won't be used if a seperate Validation Data Folder is being passed
es_patience = 5             # Early Stopping Patience

#--------------------------------------------------------------------------------------------------------------------------------
# Artifacts (Directory names to store the results & resources, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
project_name = 'CV_pro'
artifacts_dir = "Artifacts"
model_dir = "Models"
plots_dir = "Plots"
model_name = "my_test_model"
experiment_name = "light model testing"
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
comments = "making comparision for optimizers"
executed_by = 'hasnain'

#--------------------------------------------------------------------------------------------------------------------------------
# Database Integration
#--------------------------------------------------------------------------------------------------------------------------------
# Please Note that before making any change in this section, create a .env file and store the mongo db connection string or MySQL credentials in the environment variables 
# Guideline for creating .env is available on project description main page
db_integration_mysql = False
db_integration_mongodb = False 