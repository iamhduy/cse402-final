import os
import glob
from sklearn.model_selection import KFold
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import functional
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

IMG_DIR = './Celebrity Faces Dataset'
NUM_FOLDS = 10
NUM_IMAGES_PER_CLASS = 100
NUM_TEST_IMAGES_PER_CLASS = NUM_IMAGES_PER_CLASS // NUM_FOLDS  # 10
NUM_CLASSES = 17
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running model on: {DEVICE}")

# Load pre-trained AlexNet
# alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
# feature_extractor = nn.Sequential(*list(alexnet.children())[:-1], nn.Flatten()).to(DEVICE)
efficient = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(
    efficient.features,
    efficient.avgpool,
    nn.Flatten()
).to(DEVICE)
feature_extractor.eval()

# Applies modified transformations to create variations for training data
preprocess = transforms.Compose([
    transforms.Resize(256),
    # transforms.CropCenter(224)  # focus on the center of the img (square of 224)
    transforms.Lambda(lambda img: functional.crop(  # Improvement (using data augmentation)
        img,
        top=8,  # Move the crop a little bit up (to match with the faces position in the img)
        left=16,  # Keep the horizontal center
        height=224,
        width=224
    )),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extractFeatures(file_paths):
    """
    Extract features from image in a list of images path
    """
    features = []

    # Process images in batches
    for path in file_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_t = preprocess(img)
            img_t = img_t.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device

            with torch.no_grad():
                # Pass through the feature extractor
                output = feature_extractor(img_t)

            features.append(output.cpu().numpy().flatten())
        except Exception as e:
            print(f"Error processing image {path}: {e}")

    return np.array(features)


def prepareAndRunData(img_dir):
    """
    Splits the face dataset into 10 folds
    """
    class_dict = {}
    class_names = []

    for class_folder in sorted(os.listdir(img_dir)):
        class_path = os.path.join(img_dir, class_folder)
        if os.path.isdir(class_path):
            class_names.append(class_folder)
            image_paths = sorted(glob.glob(os.path.join(class_path, '*.jpg')))
            class_dict[class_folder] = np.array(image_paths)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=False)
    indices = np.arange(NUM_IMAGES_PER_CLASS)

    accuracy_results = []
    true_labels = []
    predicted_labels = []

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(indices)):
        fold_num = fold_idx + 1
        print(f"\nfold {fold_num}/{NUM_FOLDS}")
        print(test_indices)
        # Lists to store the file paths for the current fold
        train_files = []
        train_labels = []
        test_files = []
        test_labels = []

        # Iterate over all 17 classes
        for class_name in class_names:
            class_paths = class_dict[class_name]

            # Test Set (10 images per class)
            current_test_paths = class_paths[test_indices]
            test_files.extend(current_test_paths)
            test_labels.extend([class_name] * NUM_TEST_IMAGES_PER_CLASS)

            # Training Set (90 images per class)
            current_train_paths = class_paths[train_indices]
            train_files.extend(current_train_paths)
            train_labels.extend([class_name] * (NUM_IMAGES_PER_CLASS - NUM_TEST_IMAGES_PER_CLASS))

        # Sanity Check
        print(f"fold {fold_num} training Set Size: {len(train_files)} images")
        print(f"fold {fold_num} testing Set Size: {len(test_files)} images")

        print("extracting training features")
        x_train = extractFeatures(train_files)
        y_train = np.array(train_labels)

        print("extracting testing features")
        x_test = extractFeatures(test_files)
        y_test = np.array(test_labels)

        # model = SVC(kernel='linear', C=1.0, random_state=42)
        model = LinearSVC(C=1.0, max_iter=10000)
        print("training SVM classifier")
        model.fit(x_train, y_train)

        print("testing classifier")
        y_pred = model.predict(x_test)

        # Calculate and store results for this fold
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracy_results.append(fold_accuracy)

        true_labels.extend(y_test)
        predicted_labels.extend(y_pred)

        print(f"fold {fold_num} accuracy: {fold_accuracy * 100:.2f}%")

    overall_accuracy = np.mean(accuracy_results)

    # Generate the overall confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)

    return overall_accuracy, cm, class_names


# Execute the function
overall_acc, conf_matrix, labels = prepareAndRunData(IMG_DIR)
print(f"\nMean Accuracy: {overall_acc * 100:.2f}%")

# Display confusion mat
if conf_matrix is not None:
    # Set the size of the plot
    plt.figure(figsize=(15, 12))

    # Draw the confusion matrix using seaborn
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
