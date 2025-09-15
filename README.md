Image Forgery Detection using Deep Learning

This project presents a robust and scalable solution for detecting image splicing forgeries using a deep learning approach. It combines pre-processing techniques with a sophisticated Convolutional Neural Network (CNN) architecture to achieve high-performance classification.

---

## Run Locally

Clone the project

```bash
git clone https://github.com/sonalm3214/Image-Forgery-Detection-using-Deep-Learning
```

Go to the project directory

```bash
cd Image_Forgery_Detection
```


How to run the Application
Open the main_app.py and run the followind command in the Terminal
```bash
streamlit run main_app.py  ```
``` 
---

Then go to:
```bash
localhost:8501
```

---
Dataset

The dataset is split into three categories:

- Train: 4498 Authentic (Class 0), 965 Tampered (Class 1)
- Validation: 964 Authentic (Class 0), 206 Tampered (Class 1)
- Test: 965 Authentic (Class 0), 208 Tampered (Class 1)

The dataset is structured as follows:

```
train_val_test_split_384x384/
│── train/
│   ├── au/   # Authentic Images
│   ├── tp/   # Tampered Images
│
│── val/
│   ├── au/
│   ├── tp/
│
│── test/
│   ├── au/
│   ├── tp/
```




---

Model Architecture

This model uses transfer learning with a VGG19 backbone, pretrained on ImageNet, to extract features from images. The final classification layers are customized for splicing forgery detection.

- Feature Extraction: Uses Error Level Analysis (ELA) to highlight manipulated regions.
- Pretrained Backbone: VGG19 (transfer learning from ImageNet).
- Fully Connected Layers: Batch normalization and dropout (0.5) for regularization.
- Optimizer: Adam for stable training.
- Loss Function: Focal Loss to handle class imbalance.

Model Summary

Input: (384, 384, 3)
VGG19 Backbone (Pretrained on ImageNet, Transfer Learning)
Fully Connected Layers (With BatchNorm & Dropout)
Dropout (0.5)
Output: Sigmoid Activation (Binary Classification)


Training & Optimization

Optimizer Choices: AdamW(Recommended), RMSprop, SGD with Momentum
Loss Functions Explored:
  - Binary Cross-Entropy (BCE)
  - Focal Loss (for imbalanced classes)
  - Dice Loss (for better foreground-background separation)
 Callbacks: Early Stopping, ReduceLROnPlateau, Model Checkpointing



Results & Evaluation  

The model was evaluated on the test set, achieving the following performance:  

|     Metric    |   Score    |
|---------------|------------|
|   Accuracy    |   96.20%   |
|   Precision   |   89.00%   |
|   Recall      |   89.42%   |
|   F1-Score    |   89.21%   |

### Confusion Matrix  
The confusion matrix helps in understanding model performance in terms of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN):
| Actual \ Predicted  | 0 (Authentic) | 1 (Tampered) |
|---------------------|---------------|--------------|
|    0 (Authentic)    |     952       |    24        |
|    1 (Tampered)     |     16        |    192       |

- True Negative (TN): Authentic image correctly classified as authentic.  
- False Positive (FP): Authentic image misclassified as tampered.  
- False Negative (FN): Tampered image misclassified as authentic.  
- True Positive (TP): Tampered image correctly classified as tampered. 
---
##  Future Improvements  

- Enhance noise analysis using PRNU-based techniques.
- Integrate attention mechanisms for improved localization of forged regions.  
- Explore contrastive learning for improved feature representation.
- Leverage Fourier Transform Analysis for frequency-based anomaly detection.  
- Incorporate Residual Pixel Analysis (RPA) with Error Level Analysis (ELA).  
- Refine splicing detection using edge and boundary irregularities.
- Improve model robustness by analyzing lighting and shadow inconsistencies. 



