"""
Created on Tue Dec 19 08:48:36 2023

@author: rramani
"""
import copy
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import time

# Defining a custom Dataset class

class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        
        self.class_to_label = {cls:i for i,cls in enumerate(self.classes)}
        
        self.label_to_class = {i:cls for cls,i in self.class_to_label.items()}
        
        self.data = self.load_data()
        
    def load_data(self):
        data = []
        
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir,class_name)
            class_label = self.class_to_label[class_name]
            
            for file_name in os.listdir(class_path):
                image_path = os.path.join(class_path, file_name)
                data.append((image_path, class_label))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
        


# Function to train and evaluate the model

def train_and_eval(custom_model, train_loader, val_loader, optimizer, criterion, epochs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    custom_model.to(device)
    
    train_losses =[]
    val_losses = []
    best_val_acc = 0.0
    gt_labels_list =[]
    
    for epoch in range(epochs):
        
        # Training phase
        custom_model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs,labels in tqdm(train_loader, desc = f' Epochs {epoch+1}/{epochs} - Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Core training steps
            optimizer.zero_grad()
            main_output,aux_output = custom_model(inputs)
            loss = criterion(main_output, labels)
            loss.backward()
            optimizer.step()

            # Performance report
            running_train_loss += loss.item()
            _, predicted = main_output.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        train_loss = running_train_loss / len(train_loader)
        train_acc = 100 * correct_train/ total_train
        train_losses.append(train_loss)


        # Validation phase
        custom_model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        predictions_list = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader , desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                gt_labels_list.extend(labels.cpu().numpy())
            
                main_output = custom_model(inputs)
                main_output_probs = F.softmax(main_output, dim =1)
            
                loss = criterion(main_output_probs, labels)
                
                running_val_loss += loss.item()
                _, predicted = main_output.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
            
                #Add to predictions list
                predictions_list.extend(main_output_probs.cpu().numpy())
            
        gt_labels = np.array(gt_labels_list)
        predictions = np.array(predictions_list)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100.0 * correct_val / total_val
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - "
               f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
               f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
         
        #empty the gt_list
        gt_labels_list.clear()    
            
        # Save the model with the best accuracy
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_weights = copy.deepcopy(custom_model.state_dict())
    
    # Load best model
    custom_model.state_dict(best_model_weights)
    
    return gt_labels, predictions, best_val_acc
        

# Start timer
start_time = time.time()

# Directories
save_dir = 'C:\\Users\\PATH\\Nova Murine CNN'

os.makedirs(save_dir, exist_ok=True)

root_dir = 'D:\\PhD\\PATH\\Training'

# Define image preprocessing
preprocess = transforms.Compose([transforms.Resize((299,299), interpolation = Image.BICUBIC),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean =[0.5, 0.5, 0.5], std =[0.25, 0.25, 0.25])
                                 ])

# Create Dataset object
custom_dataset = Dataset(root_dir, transform = preprocess)

# Get all files and labels
all_files = [item[0] for item in custom_dataset.data]
all_labels = [item[1] for item in custom_dataset.data]

# Convert to numpy arrays for easy indexing
all_files = np.array(all_files)
all_labels = np.array(all_labels)

# Dataloaders
loader = DataLoader(custom_dataset, batch_size = 32, shuffle = True)

# Loss function
criterion = nn.CrossEntropyLoss()

# Range of hyperparameters
epochs_range = [5, 10, 15]
learning_rates = [0.001, 0.01, 0.1]

# All results list
all_results = []

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

for fold, (train_index, val_index) in enumerate(skf.split(all_files,all_labels)):
    train_dataset = torch.utils.data.Subset(custom_dataset, train_index)
    val_dataset = torch.utils.data.Subset(custom_dataset, val_index)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    for epochs in epochs_range:
        for lr in learning_rates:
            
            # Load model
            custom_model = models.inception_v3(pretrained=True)
            
            # Modify for transfer learning
            num_ftrs = custom_model.fc.in_features
            custom_model.fc = nn.Linear(num_ftrs,3)
            
            # Add Stochastic Gradient Descent with Momentum 0.9
            optimizer = optim.SGD(custom_model.parameters(), lr=lr, momentum=0.9)
            
            # Run the big training and evaluation function
            gt_labels, predictions, accuracy = train_and_eval(custom_model, train_loader, val_loader, optimizer, criterion, epochs)
            
            # Save model
            model_filename = f'{save_dir}/fold{fold}_epoch_{epochs}_lr{lr}_model.pth'
            torch.save(custom_model, model_filename)
            
            print(f'Accuracy: {accuracy}')
            
            # Convert raw model outputs to class labels
            gt_labels_t = [torch.tensor(gt).item() for gt in gt_labels]
            predicted_labels = [torch.argmax(torch.tensor(pred)).item() for pred in predictions]
            
            # Compute confusion matrix
            conf_matrix = confusion_matrix(gt_labels_t, predicted_labels)

            # Extract TP, FP, FN, TN for each class
            tp = np.diag(conf_matrix)
            fp = np.sum(conf_matrix, axis=0) - tp
            fn = np.sum(conf_matrix, axis=1) - tp
            tn = np.sum(conf_matrix) - (tp + fp + fn)

            # Compute metrics for each class
            acc = ((tp+tn)/(tp+tn+fp+fn))*100
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            recall = sensitivity  
            f1 = 2 * (precision * recall) / (precision + recall)
            
            
            results_row = {'model name': model_filename,
                           'fold': fold,
                           'epochs': epochs,
                           'lr': lr,
                           'ND TP': tp[0],
                           'ND FP': fp[0],
                           'ND TN': tn[0],
                           'ND FN': fn[0],
                           'ND accuracy': acc[0],
                           'ND sensitivity - recall': sensitivity[0],
                           'ND specificity': specificity[0],
                           'ND precision': precision[0],
                           'ND recall': recall[0],
                           'ND f1 score': f1[0],
                           'LG TP': tp[1],
                           'LG FP': fp[1],
                           'LG TN': tn[1],
                           'LG FN': fn[1],
                           'LG accuracy': acc[1],
                           'LG sensitivity': sensitivity[1],
                           'LG specificity': specificity[1],
                           'LG precision': precision[1],
                           'LG recall': recall[1],
                           'LG f1 score': f1[1],
                           'HG TP': tp[2],
                           'HG FP': fp[2],
                           'HG TN': tn[2],
                           'HG FN': fn[2],
                           'HG accuracy': acc[2],
                           'HG sensitivity': sensitivity[2],
                           'HG specificity': specificity[2],
                           'HG precision': precision[2],
                           'HG recall': recall[2],
                           'HG f1 score': f1[2]
                           }
            
            all_results.append(results_row)

results_df = pd.DataFrame(all_results)


results_df.to_excel('C:\\Users\\PATH\\custom_training_results.xlsx', index=False)

# End timer
end_time = time.time()

elapsed = end_time - start_time

print(f"Time taken for training all models is {elapsed} seconds")
