# COVID-19 Detection Using Vision Transformer and K-Fold Cross Validation

## Project Overview

This project aims to develop an efficient COVID-19 detection model utilizing the Vision Transformer (ViT) architecture and implementing K-Fold cross-validation. The dataset consists of chest X-ray images, and the model is trained on augmented data for increased robustness.

## Setup and Dependencies

1. **Google Colab Setup:**
   - The project is designed to run on Google Colab. Ensure that your Google Drive is mounted to access the dataset.

2. **Libraries Used:**
   - The following Python libraries are required for the project:
     - torch
     - torchvision
     - scikit-learn
     - transformers
     - imgaug
     - seaborn
     - matplotlib

   Install these libraries using the following command:

   ```bash
   !pip install torch torchvision scikit-learn transformers imgaug seaborn matplotlib
# Notebook Structure

## Data Loading and Augmentation

1. **Mount Google Drive:**
   - Access the dataset by mounting Google Drive.

2. **Import Libraries:**
   - Import necessary libraries, including Torch, torchvision, scikit-learn, transformers, imgaug, seaborn, and matplotlib.

3. **Set Device to CUDA:**
   - If available, set the device to CUDA for GPU acceleration.

4. **Define Augmentation Sequence:**
   - Create an augmentation sequence using imgaug to increase data diversity.

5. **Apply Data Transformations:**
   - Use torchvision's `transforms.Compose` and a custom function to apply data transformations.

## Dataset Splitting

- Split the dataset into training (80%) and testing (20%) sets using `random_split` from Torch's DataLoader.

## K-Fold Cross-Validation Setup

1. **Implement 5-Fold Cross-Validation:**
   - Use StratifiedKFold to implement 5-fold cross-validation.

2. **Define ViT Model:**
   - Utilize ViTFeatureExtractor and ViTForImageClassification from the transformers library.

3. **Move Model to GPU:**
   - Ensure the model is moved to the GPU for efficient training.

## Training

1. **Set up Loss and Optimizer:**
   - Configure the loss criterion (CrossEntropyLoss) and optimizer (Adam).

2. **Train Model:**
   - Train the model for each fold using the training dataset.

## Testing

1. **Evaluate Model:**
   - Assess the model on the test dataset for each fold.

2. **Generate Performance Metrics:**
   - Calculate accuracy scores, classification reports, and confusion matrices.

## Results Visualization

- Visualize the model's performance using seaborn and matplotlib.
- Display the confusion matrix for each fold.

## Usage

### Google Colab

1. Open the notebook in Google Colab.
2. Run each cell sequentially.

### Customization

- Adjust hyperparameters, model configurations, or augmentation settings as needed.

## Note

- Ensure GPU support is enabled in Colab for faster training.
- Customize paths and filenames according to your dataset structure.

## Results

- Save the trained model weights, confusion matrices, and any relevant visualizations.
- Document the obtained results for future reference.
