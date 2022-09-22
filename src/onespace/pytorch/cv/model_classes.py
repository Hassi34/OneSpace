from re import A, T
import torch 
import torch.nn.functional as F
import torch.nn as nn 
from .model_zoo import download_base_model

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc, 'labels': labels, 'predictions': out} 
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, epochs, result):
        print(f"Epoch: [{epoch+1}/{epochs}] ==> train_loss: {result['train_loss']:.4f} val_loss: {result['val_loss']:.4f} val_accuracy: { result['val_acc']:.4f} initial_lr: {result['lrs'][0]:.5f}  final_lr: {result['lrs'][-1]:.5f}")


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

class TransferLearningEntry(ImageClassificationBase):
    def __init__(self, MODEL_ARCHITECTURE, NUM_CLASSES):
        super().__init__()
        # Use a pretrained model
        self.network = download_base_model(MODEL_ARCHITECTURE)
        for param in self.network.parameters():
            param.requires_grad = False
        # Replace last layer
        try:
            self.network.fc = nn.Linear(self.network.fc.in_features, NUM_CLASSES)
        except AttributeError:
            try:
                input_features = self.network.classifier[6].in_features
                self.network.classifier[6] = nn.Linear(input_features, NUM_CLASSES)
            except IndexError:
                try:
                    input_features = self.network.classifier[2].in_features
                    self.network.classifier[2] = nn.Linear(input_features, NUM_CLASSES)
                except:
                    try:
                        input_features = self.network.classifier[1].in_features
                        self.network.classifier[1] = nn.Linear(input_features, NUM_CLASSES)
                    except:
                        try:
                            input_features = self.network.classifier[3].in_features
                            self.network.classifier[3] = nn.Linear(input_features, NUM_CLASSES)
                        except AttributeError:
                            self.network.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1)) # SqeezeNet
            except AttributeError:
                input_features = self.network.heads[0].in_features
                self.network.heads[0] = nn.Linear(input_features, NUM_CLASSES)
            except TypeError:
                input_features = self.network.classifier.in_features
                self.network.classifier = nn.Linear(input_features, NUM_CLASSES)
                
        except Exception as e :
            raise e

    def forward(self, xb):
        return self.network(xb)