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
from tkinter import Tk, filedialog


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Dynamically retrieve class names
        self.class_to_label = {cls: i for i, cls in enumerate(self.classes)}
        self.label_to_class = {i: cls for cls, i in self.class_to_label.items()}
        self.data = self.load_data()

    def load_data(self):
        data = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_label = self.class_to_label[class_name]
            for file_name in os.listdir(class_path):
                image_path = os.path.join(class_path, file_name)
                data.append((image_path, class_label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, os.path.basename(image_path)


def testing_all_models():
    # Tkinter dialog for generalization
    Tk().withdraw()

    # Prompt for directories and file paths
    models_dir = filedialog.askdirectory(title="Select Directory with Saved Models")
    if not models_dir:
        print("No models directory selected. Exiting.")
        return

    root_dir = filedialog.askdirectory(title="Select Testing Dataset Directory")
    if not root_dir:
        print("No dataset directory selected. Exiting.")
        return

    results_file = filedialog.asksaveasfilename(
        title="Select Location to Save Results",
        defaultextension=".xlsx",
        filetypes=[("Excel Files", "*.xlsx")]
    )
    if not results_file:
        print("No results file location selected. Exiting.")
        return

    # Prompt for specific models
    specific_models = filedialog.askopenfilenames(
        title="Select Specific Models to Log Predictions",
        initialdir=models_dir,
        filetypes=[("PyTorch Model Files", "*.pth *.pt")]
    )
    if not specific_models:
        print("No specific models selected. Continuing without specific model logging.")

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((299, 299), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])

    # Create Dataset and DataLoader
    test_dataset = CustomDataset(root_dir, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize variables
    models_list = os.listdir(models_dir)
    all_results = []
    specific_model_results = {model: [] for model in specific_models}
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate through models
    for model in models_list:
        model_path = os.path.join(models_dir, model)
        custom_model = torch.load(model_path)
        custom_model.to(device)
        custom_model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels, file_names in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = custom_model(inputs)
                probabilities = torch.softmax(outputs, dim=1)  # Confidence scores
                _, predicted = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                # Log predictions and confidence scores for specific models
                if model_path in specific_models:
                    for file_name, true_label, pred_label, confidence in zip(
                        file_names, labels.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy()
                    ):
                        specific_model_results[model_path].append({
                            'Image Name': file_name,
                            'True Label': test_dataset.label_to_class[true_label],
                            'Predicted Label': test_dataset.label_to_class[pred_label],
                            'Confidence Scores': confidence
                        })

            # Compute confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            tp = np.diag(conf_matrix)
            fp = np.sum(conf_matrix, axis=0) - tp
            fn = np.sum(conf_matrix, axis=1) - tp
            tn = np.sum(conf_matrix) - (tp + fp + fn)

            acc = ((tp + tn) / (tp + tn + fp + fn)) * 100
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            recall = sensitivity
            f1 = 2 * (precision * recall) / (precision + recall)

            # Dynamically handle multiple classes based on directory names
            results_row = {'model name': model}
            for i, class_name in enumerate(test_dataset.classes):
                results_row.update({
                    f'{class_name} TP': tp[i],
                    f'{class_name} FP': fp[i],
                    f'{class_name} TN': tn[i],
                    f'{class_name} FN': fn[i],
                    f'{class_name} accuracy': acc[i],
                    f'{class_name} sensitivity': sensitivity[i],
                    f'{class_name} specificity': specificity[i],
                    f'{class_name} precision': precision[i],
                    f'{class_name} recall': recall[i],
                    f'{class_name} f1 score': f1[i],
                })
            print(results_row)
            all_results.append(results_row)

    # Save aggregated results to Excel
    results_df = pd.DataFrame(all_results)
    results_df.to_excel(results_file, index=False)
    print(f"Aggregated results saved to {results_file}")

    # Save specific model predictions to separate Excel files
    for model_path, predictions in specific_model_results.items():
        model_name = os.path.basename(model_path)
        specific_model_df = pd.DataFrame(predictions)
        specific_model_output_file = os.path.join(
            os.path.dirname(results_file), f"{model_name}_predictions.xlsx"
        )
        specific_model_df.to_excel(specific_model_output_file, index=False)
        print(f"Predictions for {model_name} saved to {specific_model_output_file}")


# Run the function
if __name__ == "__main__":
    start_time = time.time()
    testing_all_models()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Time taken for testing all models is {elapsed} seconds")
