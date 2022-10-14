"""
Author : Hasanain Mehmood
Contact : hasanain@aicaliber.com 
"""

import os
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from .common import get_unique_filename, denormalize, evaluate, get_lr, download_grid, get_classwise_img_count
from .model_classes import ResNet9, TransferLearningEntry
from .model_zoo import download_base_model
from .data_management import get_data_loaders
from .param_selection import get_lr_scheduler, get_opt_func
import torch.nn.functional as F
import uuid
import datetime
import csv
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
import tensorflow as tf
from ...dbOps.mongo.mongoExe import save_logs_in_mongo
from ...dbOps.mysql.mysqlExe import save_logs_in_mysql


class Experiment:
    """ This class shall be used to train a Convolutional Neural Network and 
        save the complete logs and data associated with it
        Written by : Hasnain
    """
    def __init__(self, config):
        self.config = config
        self.OPTIMIZER = config.optimizer
        self.LR_SCHEDULER = config.lr_scheduler
        self.EPOCHS = config.epochs
        self.VALIDATION_SPLIT = config.validation_split
        self.IMAGE_SIZE = config.input_shape
        self.BATCH_SIZE = config.batch_size
        self.DATA_AUGMENTATION = config.augmentation
        self.MODEL_ARCHITECTURE = config.model_architecture
        self.WEIGHT_DECAY = config.weight_decay
        self.GRAD_CLIP = config.grad_clip

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
            Written by : Hasanain
        """
        self.parent_dir = os.path.join("ComputerVision", "PyTorch", self.project_name) 
        if self.train_folder_name is not None:
            self.train_dir = os.path.join(self.data_dir ,self.train_folder_name)
            self.val_dir = os.path.join(self.data_dir ,self.val_folder_name)
        else: 
            self.train_dir = None 
            self.val_dir = None
        (self.train_dl, self.val_dl, self.training_images_count,
         self.val_images_count, self.NUM_CLASSES, self.class_names) = get_data_loaders(
                                                                                        self.VALIDATION_SPLIT,
                                                                                        self.IMAGE_SIZE,
                                                                                        self.BATCH_SIZE,
                                                                                        self.data_dir,
                                                                                        self.DATA_AUGMENTATION,
                                                                                        self.train_dir, 
                                                                                        self.val_dir)  
        self.print_classwise_img_count()
        self.get_tensorboard_writer()
        self.imgs_to_tensorboard()
        self.save_image_grid()
        if self.MODEL_ARCHITECTURE == 'Inception_v3' and self.IMAGE_SIZE[1] < 299 :
            print("!!! Unsuccessful Execution !!!")
            print(f" Error : The original Inception model expects an input with minimum of size (299, 299) while {(self.IMAGE_SIZE[0], self.IMAGE_SIZE[1])} has been provided")
            sys.exit()

        if self.TransferLearning:
            self.base_model = download_base_model(self.MODEL_ARCHITECTURE)
            self.save_base_model()
            print("*****" * 13)
            print("Base Model Summary")
            print("*****" * 13)
            print(self.base_model)
            self.custom_model = TransferLearningEntry(self.MODEL_ARCHITECTURE, self.NUM_CLASSES)
            print("\n")
            print("*****" * 13)
            print("Custom Model Summary")
            print("*****" * 13)
            print(self.custom_model)

            self.opt_func = get_opt_func(self.OPTIMIZER, self.custom_model.parameters(),weight_decay = self.WEIGHT_DECAY)
            self.lr_schedule = get_lr_scheduler(self.LR_SCHEDULER , self.opt_func, self.EPOCHS,
                                                steps_per_epoch = len(self.train_dl))
            trained_model = self.fit_model()
            self.save_final_model(trained_model)
            self.save_plot()
        else:
            self.custom_model = ResNet9(self.IMAGE_SIZE[-1], self.NUM_CLASSES)
            print("\n")
            print("*****" * 13)
            print("Custom model summary")
            print("*****" * 13)
            print(self.custom_model)

            self.opt_func = get_opt_func(self.OPTIMIZER, self.custom_model.parameters(),weight_decay = self.WEIGHT_DECAY)
            self.lr_schedule = get_lr_scheduler(self.LR_SCHEDULER , self.opt_func,
                                                self.EPOCHS, steps_per_epoch = len(self.train_dl))
            
            trained_model = self.fit_model()
            self.save_final_model(trained_model)
            self.save_plot()

    def fit_model(self):
        """This method will perform the operation on data and model architecture
            and will provide the trained model with call backs
           Written by : Hasnain
        """ 
        torch.cuda.empty_cache()
        class_labels = []
        class_preds = []
        self.history = []
        start_time = timeit.default_timer()
        global_step = 0
        for epoch in range(self.EPOCHS):
            # Training Phase
            self.custom_model.train()
            train_losses = []
            lrs = []
            
            for batch in tqdm(self.train_dl):
                loss = self.custom_model.training_step(batch)

                self.writer.add_scalar("Training Loss", loss, global_step = global_step)
                val_resut = self.custom_model.validation_step(batch)
                class_probs_batch = [F.softmax(output, dim=0) for output in val_resut['predictions']]
                class_preds.append(class_probs_batch)

                class_labels.append(val_resut['labels'])
                self.writer.add_scalar("Validation Loss", val_resut['val_loss'], global_step = global_step)
                self.writer.add_scalar("Validation Accuracy", val_resut['val_acc'], global_step = global_step)
                global_step += 1
                
                train_losses.append(loss)
                loss.backward()

                # Gradient clipping
                if not isinstance (self.GRAD_CLIP, type(None)):
                    nn.utils.clip_grad_value_(self.custom_model.parameters(), self.GRAD_CLIP)

                self.opt_func.step()
                self.opt_func.zero_grad()

                # Record & update learning rate
                learning_rate = get_lr(self.opt_func)
                self.writer.add_scalar("Learning Rate", learning_rate, global_step = global_step)
                lrs.append(learning_rate)
                self.lr_schedule.step()
                global_step += 1
            
                
            # Validation phase
            result = evaluate(self.custom_model, self.val_dl)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            self.writer.add_hparams({'final_lr': lrs[-1], 'bsize': self.BATCH_SIZE, 'epoch': epoch+1},
                                    {'val_accuracy': result['val_acc'], 'val_loss': result['val_loss']})
            self.custom_model.epoch_end(epoch, self.EPOCHS, result)
            self.history.append(result)
        end_time = timeit.default_timer()
        self.training_time = round((end_time - start_time)/60.0, 3)
        images, batch_labels = next(iter(self.train_dl))
        self.writer.add_graph(self.custom_model, images)

        class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
        class_labels = torch.cat(class_labels)
        for i in range(self.NUM_CLASSES):
            labels_i = class_labels == i
            preds_i = class_preds[:, i]
            self.writer.add_pr_curve(str(i), labels_i, preds_i, global_step=global_step)

        batch_labels = [self.train_dl.dl.dataset.classes[label] for label in batch_labels] 
        features = images.reshape(images.shape[0], -1)
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
        self.writer.add_embedding(features, metadata = batch_labels, label_img = images, global_step = 0)
        self.writer.close()
        return self.custom_model
    def print_classwise_img_count(self):
        """
        Prints the number of images for every class in the dataset
        """
        train_data = get_classwise_img_count(self.train_dl)
        val_data = get_classwise_img_count(self.val_dl)
        print("*****" * 13)
        print("Class wise count of Images")
        print("*****" * 13)
        print(f"\n==> Training Data : {train_data}\n")
        print(f"\n==> Validation Data : {val_data}\n")

    def save_final_model(self, model):
        """This method with create the "models" directory
            and will save trained model in that
            Written by : Hasnain
        """
        model_dir_path = os.path.join(self.parent_dir, self.artifacts_dir,self.model_dir, 'TrainedModels')
        os.makedirs(model_dir_path, exist_ok = True)
        unique_filename = get_unique_filename(self.model_name, is_model_name=True)
        path_to_model = os.path.join(model_dir_path, unique_filename)
        torch.save(model.state_dict(), path_to_model)
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
        torch.save(self.base_model.state_dict(), path_to_model)
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
        self.history = pd.DataFrame(self.history)
        self.history.plot(figsize= (8,5))
        print("\n")
        print("*****" * 13)
        print("Model Performance Metrics")
        print("*****" * 13)
        print(self.history)
        
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.savefig(path_to_plot_result)
        print("\n")
        print("*****" * 13)
        print("Model performance mertics plot saved at the following location")
        print("*****" * 13)
        print(f"\n ==> {path_to_plot_result}\n")
        self.record_logs()

    def record_logs(self):
        logs_header = ['Experiment ID','Exeriment Name', 'Executed By', 'Local Date Time','UTC Date Time', 'Optimizer',
        'Epochs', '% Validation Split','Image Dimensions','Batch Size', 'Data Augmentation', 'Model Architechture',
        'Training Images Count', 'Validation Iamges Count','Number of Classes', 
        'Class Names', 'Gradient Clipping', 'Weight Decay']  + self.history.columns.tolist() + ['Training Time(Minutes)','Comments']
        
        history_values = [self.history[col_name].tolist()[-1] for col_name in self.history.columns.tolist()]
        
        logs_data = [str(uuid.uuid4()),self.experiment_name, self.executed_by, datetime.datetime.now(), datetime.datetime.utcnow(),
        self.OPTIMIZER,self.EPOCHS, (self.VALIDATION_SPLIT)*100, f"{self.IMAGE_SIZE}", self.BATCH_SIZE,
         self.DATA_AUGMENTATION, self.MODEL_ARCHITECTURE, self.training_images_count, self.val_images_count,
        self.NUM_CLASSES, str(tuple(self.class_names)), self.GRAD_CLIP, f"{self.WEIGHT_DECAY}"] + history_values + [self.training_time, self.comments] 
    
        self.csv_logs_dir = os.path.join(self.parent_dir, self.logs_dir, self.csv_logs_dir_name)
        os.makedirs(self.csv_logs_dir, exist_ok=True)
        csv_logs_file = os.path.join(self.csv_logs_dir, self.csv_logs_file)
        
        with open(csv_logs_file, 'a') as logs_file:
            writer = csv.writer(logs_file, lineterminator='\n')
            if os.stat(csv_logs_file).st_size == 0:
                writer.writerow(logs_header)
            writer.writerow(logs_data)
        logs_data[-3] = str(logs_data[-3])
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
        print(f"\n************* Kudos, Experiment compeleted successfully! ************\n")
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
        download_grid(self.train_dl, imagenet_stats = self.imagenet_stats, image_file_path= path_to_train_img_grid)
    
        val_img_dir = os.path.join(self.parent_dir, self.artifacts_dir, "Images", "Validation Images")
        os.makedirs(val_img_dir, exist_ok=True)

        path_to_val_img_grid = os.path.join(val_img_dir, unique_filename_result)
        download_grid(self.val_dl, imagenet_stats = self.imagenet_stats, image_file_path= path_to_val_img_grid)
        print("\n")
        print("*****" * 13)
        print("Training and Validation Images have been saved at the following locations respectively:")
        print("*****" * 13)
        print(f"\n ==> {path_to_train_img_grid}\n")
        print(f"\n ==> {path_to_val_img_grid}\n")


    def imgs_to_tensorboard(self):
        self.imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        train_images, labels = next(iter(self.train_dl))
        train_images = denormalize(train_images, *self.imagenet_stats)
        train_grid = torchvision.utils.make_grid(train_images)

        val_images, labels = next(iter(self.val_dl))
        val_images = denormalize(val_images, *self.imagenet_stats)
        val_grid = torchvision.utils.make_grid(val_images)

        self.writer.add_image('Training Images', train_grid, 0)
        self.writer.add_image('Validation Images', val_grid, 0)
    def get_tensorboard_writer(self):
        try: self.tensorboard_logs_dir
        except AttributeError: 
            folder_name = get_unique_filename("logs")
            self.tensorboard_logs_dir = os.path.join(self.parent_dir, self.logs_dir, self.tensorboard_root_log_dir, folder_name)
            os.makedirs(self.tensorboard_logs_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_logs_dir)
        