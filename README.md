# 🛡️ Adversarial Training on CIFAR-10 with ResNet18

This project demonstrates how to make a deep learning model more robust against adversarial attacks using **adversarial training**. The task is image classification on the **CIFAR-10** dataset, and the attack used is **FGSM (Fast Gradient Sign Method)**.

---

## 📦 Dataset

- **Name:** CIFAR-10  
- **Description:** 60,000 32x32 color images in 10 classes  
- **Split:** 50,000 training images / 10,000 test images  
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 🧠 Model

- **Architecture:** ResNet18 (modified)
- **Modifications:**
  - First convolution adapted for 32×32 input images
  - MaxPooling removed
  - Final fully-connected layer adjusted for 10-class output
- **Framework:** PyTorch

---

## 💣 Adversarial Attack (FGSM)

- **Method:** Fast Gradient Sign Method (FGSM)
- **Formula:**  
  \[
  x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
  \]
- **Purpose:** Add small perturbations to input images that lead to misclassification by the model

---

## 🛠️ Adversarial Training

During training, we use both:
- Clean images  
- FGSM-generated adversarial images (with epsilon = 0.1)

The model learns to classify both correctly, improving robustness.

---

## 🧪 Evaluation

After training, the model is evaluated on both clean and adversarial test sets.

**Results (example):**

| Metric                | Value     |
|-----------------------|-----------|
| Accuracy (Clean)      | 88.27%    |
| Accuracy (Adversarial)| 58.34%    |

---

## 📂 File Structure

ai-adversarial-training-cifar10/
│
├── model_resnet.py # ResNet18 adapted for CIFAR-10
├── adversarial.py # FGSM attack function
├── main_adversarial_training.ipynb # Colab notebook for training
├── evaluate_model.ipynb # Local CPU notebook for evaluation
├── resnet18_adversarial_trained.pth # Saved trained model
└── README.md # This file

yaml
Copia
Modifica

---

## 🚀 How to Run (locally)

1. Clone the repo and install dependencies:
   ```bash
   pip install torch torchvision tqdm
Run evaluate_model.ipynb to test the trained model:

Accuracy on clean test set

Accuracy on FGSM-adversarial test set

📌 Key Learnings
Training with adversarial examples can significantly improve robustness

Even strong models like ResNet are vulnerable without defense

FGSM is a simple yet effective method to simulate attacks

🧠 Author
Emanuele Rossi – AI Red Teaming & Adversarial ML enthusiast
GitHub: @Ema-reds