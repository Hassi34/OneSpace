"""
Author : Hasnain Mehmood
Contact : hasanain@aicaliber.com 
"""
import tensorflow as tf 
import os
import timeit
import pandas as pd
import matplotlib.pyplot as plt
from .callbacks import get_callbacks
from .common import get_unique_filename, set_memory_growth, img_to_array, get_classwise_img_count
from .model_zoo import download_base_model
from .data_management import get_data_generators
from .param_selection import learning_rate_decay, get_optimizer
import uuid
import datetime
import csv
import random
from ...dbOps.mongo.mongoExe import save_logs_in_mongo
from ...dbOps.mysql.mysqlExe import save_logs_in_mysql

class Experiment:
    """ This class shall be used to train a Convolutional Neural Network and 
        save the complete logs and data associated with it
        Written by : Hasnain
    """
    def __init__(self, config):
        self.config = config
        self.ACTIVATION = config.activation
        self.ACTIVATION_OUTPUT = config.activation_output
        self.LOSS_FUNCTION = config.loss_function
        self.OPTIMIZER = config.optimizer
        self.METRICS = config.metrics
        self.EPOCHS = config.epochs
        self.ES_PATIENCE = config.es_patience
        self.VALIDATION_SPLIT = config.validation_split
        self.IMAGE_SIZE = config.input_shape
        self.BATCH_SIZE = config.batch_size
        self.DATA_AUGMENTATION = config.augmentation
        self.MODEL_ARCHITECTURE = config.model_architecture
        self.FREEZE_ALL = config.freeze_all
        self.FREEZ_TILL = config.freeze_till
        self.LR_SCHEDULER = config.lr_scheduler

        self.data_dir = os.path.join(".",config.data_dir)
        self.project_name = config.project_name
        self.TransferLearning  = config.transfer_learning
        self.model_ckpt_dir = config.model_ckpt_dir
        self.artifacts_dir = config.artifacts_dir
        self.model_dir = config.model_dir
        self.train_folder_name = config.train_folder_name 
        self.val_folder_name = config.val_folder_name
        self.model_name = config.model_name
        self.callbacked_model_name = config.callbacked_model_name
        self.logs_dir = config.logs_dir
        self.plots_dir = config.plots_dir
        self.plot_name = config.plot_name
        self.tensorboard_root_log_dir = config.tensorboard_root_log_dir
        self.csv_logs_dir_name = config.csv_logs_dir
        self.csv_logs_file = config.csv_logs_file
        self.experiment_name = config.experiment_name
        self.comments = config.comments
        self.executed_by = config.executed_by
        self.db_integration_mysql = config.db_integration_mysql
        self.db_integration_mongodb = config.db_integration_mongodb

    def run_experiment(self):
        """This method will start an experiment
           with variables provided at initialization
            Written by : Hasnain
        """
        self.parent_dir = os.path.join("ComputerVision", "TensorFlow", self.project_name)
        set_memory_growth()

        if not isinstance(self.train_folder_name , type(None)):
            self.train_dir = os.path.join(self.data_dir ,self.train_folder_name)
            self.val_dir = os.path.join(self.data_dir ,self.val_folder_name)
        else: 
            self.train_dir = None 
            self.val_dir = None
        self.train_generator, self.valid_generator = get_data_generators(
                                                                        self.VALIDATION_SPLIT,
                                                                        self.IMAGE_SIZE,
                                                                        self.BATCH_SIZE,
                                                                        self.data_dir,
                                                                        self.DATA_AUGMENTATION,
                                                                        self.train_dir, 
                                                                        self.val_dir
                                                                        )

        self.NUM_CLASSES = self.train_generator.num_classes
        self.save_image_grid()
        self.print_classwise_img_count()
        if self.TransferLearning:
            self.base_model = download_base_model(self.MODEL_ARCHITECTURE, self.IMAGE_SIZE)
            self.save_base_model()
            print("*****" * 13)
            print("Base Model Summary")
            print("*****" * 13)
            self.base_model.summary()

            if self.FREEZE_ALL:
                for layer in self.base_model.layers:
                    layer.trainable = False
            elif not isinstance(self.FREEZ_TILL, type(None)):
                for layer in self.base_model.layers[:self.FREEZ_TILL]:
                    layer.trainable = False

                # add custom layers -
            flatten_in = tf.keras.layers.Flatten()(self.base_model.output)
            prediction = tf.keras.layers.Dense(
                units=self.NUM_CLASSES,
                activation=self.ACTIVATION_OUTPUT
            )(flatten_in)

            self.custom_model = tf.keras.models.Model(
                inputs=self.base_model.input,
                outputs = prediction
            )
            print("\n")
            print("*****" * 13)
            print("Custom Model Summary")
            print("*****" * 13)
            self.custom_model.summary()
            if not isinstance(self.LR_SCHEDULER, type(None)):
                lr_schedule = learning_rate_decay(self.LR_SCHEDULER)
                optimizer_with_lr_schedule = get_optimizer(self.OPTIMIZER, lr_schedule)
                self.custom_model.compile(
                loss = self.LOSS_FUNCTION,
                optimizer = optimizer_with_lr_schedule,
                metrics = self.METRICS
            )
            else :
                self.custom_model.compile(
                    loss = self.LOSS_FUNCTION,
                    optimizer = self.OPTIMIZER,
                    metrics = self.METRICS
                )
            self.custom_model = self.fit_model(self.custom_model)
            self.save_final_model(self.custom_model)
            self.save_plot()
        else:
            
            LAYERS = [
                tf.keras.layers.Flatten(input_shape = self.IMAGE_SIZE, name = "inputLayer"),
                tf.keras.layers.Dense(300, activation = self.ACTIVATION, name = "hiddenLayer1"),
                tf.keras.layers.Dense(100, activation = self.ACTIVATION, name = "hiddenLayer2"),
                tf.keras.layers.Dense(self.NUM_CLASSES, activation = self.ACTIVATION_OUTPUT, name = "OutputLayer")
            ]
            self.custom_model = tf.keras.models.Sequential(LAYERS)
            print("\n")
            print("*****" * 13)
            print("Custom model summary")
            print("*****" * 13)
            self.custom_model.summary()
            if not isinstance(self.LR_SCHEDULER, type(None)):
                lr_schedule = learning_rate_decay(self.LR_SCHEDULER)
                optimizer_with_lr_schedule = get_optimizer(self.OPTIMIZER, lr_schedule)
                self.custom_model.compile(
                    loss = self.LOSS_FUNCTION,
                    optimizer = optimizer_with_lr_schedule,
                    metrics = self.METRICS)  

            else :
                self.custom_model.compile(
                    loss = self.LOSS_FUNCTION,
                    optimizer = self.OPTIMIZER,
                    metrics = self.METRICS
                )
            
            self.custom_model = self.fit_model(self.custom_model)
            
            self.save_final_model(self.custom_model)
            self.save_plot()

    def fit_model(self, model : object) -> object:
        """This method will perform the operation on data and model architecture
            and will provide the trained model with call backs
           Written by : Hasnain
        """ 
        folder_name = get_unique_filename("logs")
        self.tensorboard_logs_dir = os.path.join(self.parent_dir, self.logs_dir, self.tensorboard_root_log_dir, folder_name)
        os.makedirs(self.tensorboard_logs_dir, exist_ok=True)

        model_ckpt_path = os.path.join(self.parent_dir, self.artifacts_dir,self.model_dir, self.model_ckpt_dir)
        os.makedirs(model_ckpt_path, exist_ok=True)
        
        early_stopping_cb, checkpointing_cb, tensorboard_cb = get_callbacks(
            self.ES_PATIENCE, self.callbacked_model_name, model_ckpt_path, self.tensorboard_logs_dir
            )

        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        start_time = timeit.default_timer()
        self.history = model.fit(
            self.train_generator,
            validation_data = self.valid_generator,
            epochs=self.EPOCHS,
            steps_per_epoch=steps_per_epoch, 
            validation_steps=validation_steps,
            callbacks = [tensorboard_cb , early_stopping_cb, checkpointing_cb]
            )
        end_time = timeit.default_timer()
        self.training_time = round((end_time - start_time)/60.0, 3)
        self.record_logs()
        return model


    def save_final_model(self, model):
        """This method with create the "models" directory
            and will save trained model in that
            Written by : Hasnain
        """
        model_dir_path = os.path.join(self.parent_dir, self.artifacts_dir,self.model_dir, 'TrainedModels')
        os.makedirs(model_dir_path, exist_ok = True)
        unique_filename = get_unique_filename(self.model_name, is_model_name=True)
        path_to_model = os.path.join(model_dir_path, unique_filename)
        model.save(path_to_model)
        print("\n")
        print("*****" * 13)
        print("Trained model saved at the following location")
        print("*****" * 13)
        print(f"\n ==> {path_to_model}\n")
        

    def save_base_model(self):
        """This method with create the "models" directory
            and will save trained model in that
            Written by : Hasnain
        """
        model_dir_path = os.path.join(self.parent_dir, self.artifacts_dir,self.model_dir, 'BaseModels')
        os.makedirs(model_dir_path, exist_ok = True)
        unique_filename = get_unique_filename(self.MODEL_ARCHITECTURE, is_model_name=True)
        path_to_model = os.path.join(model_dir_path, unique_filename)
        self.base_model.save(path_to_model)
        print("\n")
        print("*****" * 13)
        print("Base Model saved at the following location")
        print("*****" * 13)
        print(f"\n ==> {path_to_model}\n")
    
    def save_plot(self):
        """This method will create the plots of defined evaluation metrics 
           and will save the plots in "plots" directory
            Written by : Hasnain
        """

        plots_dir_path_results = os.path.join(self.parent_dir, self.artifacts_dir, self.plots_dir , "results")
        os.makedirs(plots_dir_path_results, exist_ok=True)

        unique_filename_result = get_unique_filename(self.plot_name)
        unique_filename_result = unique_filename_result+".png"
        path_to_plot_result = os.path.join(plots_dir_path_results, unique_filename_result)
        
        plots_dir_path_models = os.path.join(self.parent_dir, self.artifacts_dir, self.plots_dir , "models")
        os.makedirs(plots_dir_path_models, exist_ok=True)

        unique_filename_model = get_unique_filename(self.model_name)
        unique_filename_model = unique_filename_model+".png"
        path_to_plot_model = os.path.join(plots_dir_path_models, unique_filename_model)
        
        pd.DataFrame(self.history.history).plot(figsize= (8,5))
        print("\n")
        print("*****" * 13)
        print("Model Performance Metrics")
        print("*****" * 13)
        print(pd.DataFrame(self.history.history))
        
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.savefig(path_to_plot_result)
        print("\n")
        print("*****" * 13)
        print("Model performance mertics plot saved at the following location")
        print("*****" * 13)
        print(f"\n ==> {path_to_plot_result}\n")
        
        try:
            tf.keras.utils.plot_model(self.custom_model, path_to_plot_model, show_shapes=True, show_layer_names=True)
            print("\n")
            print("*****" * 13)
            print("Model plot saved at the following location")
            print("*****" * 13)
            print(f"\n ==> {path_to_plot_model}\n")
        except ImportError:
            print("\n")
            print("*****" * 13)
            print("Run the following 2 commands on your terminal to get the model plot")
            print("*****" * 13)
            print(f"\n==> conda install pydot\n==> conda install pydotplus\n")
            print(f"!! Alert !!: Skipping the model plot for the current run...\n")
        except Exception as e:
            raise e
        finally:
            print(f"\n************* Kudos, Experiment compeleted successfully! ************\n")
    
    def print_classwise_img_count(self):
        """
        Prints the number of images for every class in the dataset
        """
        train_data = get_classwise_img_count(self.train_generator.classes, [*self.train_generator.class_indices])
        val_data = get_classwise_img_count(self.valid_generator.classes, [*self.valid_generator.class_indices])
        print("*****" * 13)
        print("Classwise count of Images")
        print("*****" * 13)
        print(f"\n==> Training Data : {train_data}\n")
        print(f"\n==> Validation Data : {val_data}\n")
    def img_grid(self, data_generator, nrows = 4 , ncols = 6, fig_width = 20, fig_height = 12, filepath = "grid_image.png"):
        _, axs = plt.subplots(nrows= nrows, ncols=ncols, figsize=(fig_width, fig_height))
        axs = axs.flatten()
        ds = list(zip(data_generator.filepaths, data_generator.classes))
        random.shuffle(ds)
        for ax in axs :
            selection = random.choice(ds)
            img, label = selection[0], selection[1]
            img = img_to_array(img)
            ax.imshow(img)
            classes_index_switched = {y:x for x,y in self.train_generator.class_indices.items()}
            ax.title.set_text(classes_index_switched[label])
            ax.axis('off')
        plt.savefig(filepath)
    def record_logs(self):
        training_images = self.train_generator.n 
        val_images = self.valid_generator.n
        classes = tuple(self.train_generator.class_indices.keys())
        logs_header = ['Experiment ID','Exeriment Name', 'Executed By', 'Local Date Time','UTC Date Time','Within Network Activation',
         'Final Activation', 'Loss Function', 'Optimizer', 'Metrics', 'Epochs', 'ES Patience', '% Validation Split','Image Dimensions',
         'Batch Size', 'Data Augmentation', 'Model Architechture','Freeze All', 'Freeze Till', 'Training Images Count', 'Validation Iamges Count',
         'Number of Classes', 'Class Names']  + list(self.history.history.keys()) + ['Training Time(Minutes)','Comments']
        
        history_values = [round(value[-1],4) for value in self.history.history.values()]
        
        logs_data = [str(uuid.uuid4()),self.experiment_name, self.executed_by, datetime.datetime.now(), datetime.datetime.utcnow(),
         self.ACTIVATION, self.ACTIVATION_OUTPUT, self.LOSS_FUNCTION, self.OPTIMIZER, f"{self.METRICS}",
         self.EPOCHS, self.ES_PATIENCE, (self.VALIDATION_SPLIT)*100, f"{self.IMAGE_SIZE}", self.BATCH_SIZE,
         self.DATA_AUGMENTATION, self.MODEL_ARCHITECTURE, self.FREEZE_ALL, self.FREEZ_TILL, training_images, val_images,
        self.NUM_CLASSES, f"{classes}"] + history_values + [self.training_time, self.comments] 
        
        self.csv_logs_dir = os.path.join(self.parent_dir, self.logs_dir, self.csv_logs_dir_name)
        os.makedirs(self.csv_logs_dir, exist_ok=True)
        csv_logs_file = os.path.join(self.csv_logs_dir, self.csv_logs_file)
        
        with open(csv_logs_file, 'a') as logs_file:
            writer = csv.writer(logs_file, lineterminator='\n')
            if os.stat(csv_logs_file).st_size == 0:
                writer.writerow(logs_header)
            writer.writerow(logs_data)

        if self.db_integration_mongodb:
            try:
                save_logs_in_mongo(self.project_name, dict(zip(logs_header, logs_data)))
            except:
                print("!!! Could not record Logs in MongoDB, Please check the connection string and premissions one again")
            finally:
                pass
        if self.db_integration_mysql:
            try:
                save_logs_in_mysql(data = logs_data, columns=logs_header, project_name=self.project_name)
            except:
                print("!!! Could not record Logs in MySQL, Please check the credentials and premissions one again")
            finally:
                pass
        print("\n")
        print("*****" * 13)
        print(f'Final CSV logs has been saved at the following location')
        print("*****" * 13)
        print(f"\n ==> {csv_logs_file}\n")

    def save_image_grid(self):
        """
        Returns:
            (str): Unique image grid name
        """
        train_img_dir = os.path.join(self.parent_dir, self.artifacts_dir, "Images", "Training Images")
        os.makedirs(train_img_dir, exist_ok=True)

        unique_filename_result = get_unique_filename("img_grid")
        unique_filename_result = unique_filename_result+".png"
        path_to_train_img_grid = os.path.join(train_img_dir, unique_filename_result)
        self.img_grid(data_generator = self.train_generator, filepath = path_to_train_img_grid)
    
        val_img_dir = os.path.join(self.parent_dir, self.artifacts_dir, "Images", "Validation Images")
        os.makedirs(val_img_dir, exist_ok=True)

        path_to_val_img_grid = os.path.join(val_img_dir, unique_filename_result)
        self.img_grid(data_generator= self.valid_generator, filepath = path_to_val_img_grid)
        print("\n")
        print("*****" * 13)
        print("Training and Validation Images have been saved at the following locations respectively:")
        print("*****" * 13)
        print(f"\n ==> {path_to_train_img_grid}\n")
        print(f"\n ==> {path_to_val_img_grid}\n")

