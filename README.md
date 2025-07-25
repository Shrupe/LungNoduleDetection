# Luna3DCNN - 3D Lung Nodule Classification with Deep Learning

A PyTorch-based 3D convolutional neural network (CNN) for classifying lung nodules from CT scan patches. This project is based on the LUNA16 dataset and performs nodule detection using volumetric (3D) image analysis.

---

## 📁 Project Structure

Luna3DCNN/
├── data/ # Preprocessed CT scan patches
│ ├── train/
│ ├── val/
│ └── test/
├── models/
│ └── luna3dcnn.py # 3D CNN model architecture
├── utils/
│ ├── preprocessing.py # Image preprocessing and lung masking
│ └── metrics.py # AUC, F1, precision, recall computations
├── train.py # Main training script
├── evaluate.py # Model evaluation and prediction
├── plot_auc.py # AUC-epoch visualization
└── README.md # Project overview and instructions

---

## 🚀 Features

- **3D CNN architecture** tailored for volumetric CT scans
- **Preprocessing**: lung segmentation using morphological operations
- **Custom dataset loader** for 3D nodule patches
- **Real-time metrics**: AUC, accuracy, F1-score, precision, recall
- **Training visualization**: AUC over epochs with `matplotlib`

---


📊 Training
Train the model on 3D lung patches:

bash
Copy
Edit
python train.py --epochs 20 --batch_size 16 --lr 1e-4
You can modify other arguments like --data_dir, --model_dir, and --log_interval.

🧠 How It Works
Input: 3D CT scan patches extracted from LUNA16 annotations

Preprocessing: Threshold + morphological lung mask segmentation

Model: 3D CNN with convolutional blocks, max pooling, and fully connected layers

Output: Sigmoid probability for nodule presence

📂 Preprocessing
Lung segmentation is performed using a custom method:

Thresholding with -320 HU

Removing background

Keeping the 2 largest connected regions (lungs)

Morphological closing

See utils/preprocessing.py for implementation.

📌 Dataset
This project uses the LUNA16 dataset, derived from the LIDC-IDRI lung cancer screening dataset.
