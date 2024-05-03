# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:25:34 2023

@author: rramani
"""

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                data.append((image_path, class_label, class_name, file_name))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        image_path, label, class_name, filename = self.data[index]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label , class_name, filename

def testing_one_model(model_name, model_path, test_set, test_loader, results_file):
    
    all_results =[]
    
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PMAC_model = torch.load(model_path)
    
    PMAC_model.to(device)
    
    PMAC_model.eval()
    
    all_results =[]
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels, class_name_tuple, filename_tuple in test_loader:
            
            class_name = class_name_tuple[0]
            filename = filename_tuple[0]
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = PMAC_model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # Apply softmax to convert logits to probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            prob_class_0 = round(probabilities[:,0].item(),2)
            prob_class_1 = round(probabilities[:,1].item(),2)
            prob_class_2 = round(probabilities[:,2].item(),2)
            prob_class_3 = round(probabilities[:,3].item(),2)
            
            
        
            # Get the actual string labels for the classes in the results
            true_label_class = test_set.label_to_class.get(labels.item())
            predicted_label_class = test_set.label_to_class.get(predicted_class)
        
        
            """
            # Uncomment these 3 statements to transfer predicted files into an output directory
            
            output_path = os.path.join(output_dir,f'predicted_as_{predicted_class}')
        
            os.makedirs(output_path, exist_ok=True)
        
            shutil.copy(image_path, output_path)
            """
            
            
            # Update the dictionary with the image information
            results_row = {'Model': model_name,
                           'Image': filename,
                           'True_Label': class_name,
                           'Predicted_Label': predicted_label_class,
                           'Prob Class 1': prob_class_0, #alphabetical order(0=HG,1=LG,2=ND)
                           'Prob Class 2': prob_class_1,
                           'Prob Class 3': prob_class_2,
                           'Prob Class 4': prob_class_3
                           }
            
            
            all_results.append(results_row)
            
        # To Pandas Dataframe
        results_df = pd.DataFrame(all_results)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
             
             
        # Extract TP, FP, FN, TN for each class
        tp = np.diag(conf_matrix)
        fp = np.sum(conf_matrix, axis=0) - tp
        fn = np.sum(conf_matrix, axis=1) - tp
        tn = np.sum(conf_matrix) - (tp + fp + fn)

        # Compute metrics for each class
        acc = np.round(((tp+tn)/(tp+tn+fp+fn))*100,2)
        sensitivity = np.round(tp / (tp + fn),2)
        specificity = np.round(tn / (tn + fp),2)
        precision = np.round(tp / (tp + fp),2)
        recall = sensitivity
        f1 = np.round(2 * (precision * recall) / (precision + recall),2)    
        
        
        metrics_row = {'True positives': tp,
                       'True negatives': tn,
                       'False positives': fp,
                       'False negatives': fn,
                       'Accuracy': acc,
                       'Sensitivity-Recall': sensitivity,
                       'Specificity': specificity,
                       'Precision': precision,
                       'F1 score': f1}
        
        metrics_df = pd.DataFrame(metrics_row)

        df = pd.concat([results_df,metrics_df], ignore_index=True)
            
        df.to_excel(results_file, index = False)
    


model_name = "custom_name_model"
model_path = "C:\\Users\\...\\model_name.pth"
root_dir= "D:\\...\\Testing"
preprocess = transforms.Compose([transforms.Resize((299, 299)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])])

MuAc_dataset = Dataset(root_dir, transform = preprocess)

test_loader = DataLoader(MuAc_dataset, shuffle=False)

results_file = "C:\\Users\\...\\best_model_test_results.xlsx"

start_time = time.time()

testing_one_model(model_name, model_path, MuAc_dataset, test_loader, results_file)
        
end_time= time.time()

elapsed = end_time - start_time

print(f" Time taken for testing one model across all test images is : {elapsed} seconds")
