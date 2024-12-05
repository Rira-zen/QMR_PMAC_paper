"""
Created on Wed Dec 20 08:46:41 2023

@author: rramani
"""


import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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



def testing_all_models(models_dir, test_loader, results_file):
    
    models_list = os.listdir(models_dir)
    
    all_results =[]
    
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    for model in models_list:
        
        custom_model = torch.load(os.path.join(models_dir,model))
        
        custom_model.to(device)
        
        custom_model.eval()
        
        y_true = []
        y_pred =[]
       
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = custom_model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
            
            # Compute confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
                
                
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
            
            results_row = {'model name': model,
                              'Class 1 TP': tp[0],
                              'Class 1 FP': fp[0],
                              'Class 1 TN': tn[0],
                              'Class 1 FN': fn[0],
                              'Class 1 accuracy': acc[0],
                              'Class 1 sensitivity - recall': sensitivity[0],
                              'Class 1 specificity': specificity[0],
                              'Class 1 precision': precision[0],
                              'Class 1 recall': recall[0],
                              'Class 1 f1 score': f1[0],
                              'Class 2 TP': tp[1],
                              'Class 2 FP': fp[1],
                              'Class 2 TN': tn[1],
                              'Class 2 FN': fn[1],
                              'Class 2 accuracy': acc[1],
                              'Class 2 sensitivity': sensitivity[1],
                              'Class 2 specificity': specificity[1],
                              'Class 2 precision': precision[1],
                              'Class 2 recall': recall[1],
                              'Class 2 f1 score': f1[1],
                              'Class 3 TP': tp[2],
                              'Class 3 FP': fp[2],
                              'Class 3 TN': tn[2],
                              'Class 3 FN': fn[2],
                              'Class 3 accuracy': acc[2],
                              'Class 3 sensitivity': sensitivity[2],
                              'Class 3 specificity': specificity[2],
                              'Class 3 precision': precision[2],
                              'Class 3 recall': recall[2],
                              'Class 3 f1 score': f1[2],
                              'Class 4 TP': tp[3],
                              'Class 4 FP': fp[3],
                              'Class 4 TN': tn[3],
                              'Class 4 FN': fn[3],
                              'Class 4 accuracy': acc[3],
                              'Class 4 sensitivity': sensitivity[3],
                              'Class 4 specificity': specificity[3],
                              'Class 4 precision': precision[3],
                              'Class 4 recall': recall[3],
                              'Class 4 f1 score': f1[3]
                              }
               
            print(results_row)
                
            all_results.append(results_row)

    results_df = pd.DataFrame(all_results)
    
    results_df.to_excel(results_file, index = False)
    
    
    
models_dir = "C:\\Users\\PATH\\saved_models"

root_dir = "D:\\PATH\\Testing"

results_file = "C:\\Users\\PATH\\all_models_test_results.xlsx"

# Define image preprocessing
preprocess = transforms.Compose([transforms.Resize((299,299), interpolation = Image.BICUBIC),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean =[0.5, 0.5, 0.5], std =[0.25, 0.25, 0.25])
                                 ])

# Create Dataset object
custom_testset = Dataset(root_dir, transform = preprocess)

# Dataloader
test_loader = DataLoader(custom_testset, batch_size = 32, shuffle = False)

start_time = time.time()

# **Actually running the main function**
testing_all_models(models_dir, test_loader, results_file)

end_time = time.time()

elapsed = end_time - start_time
print(f"Time taken for testing all models is {elapsed} seconds")
