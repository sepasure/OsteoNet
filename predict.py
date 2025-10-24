import os
import json
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import efficientnetv2_m as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    time_points = ['D0', 'D1', 'D3', 'D5', 'D7']

    weights_base_dir = "./weights/0-5D"
    test_data_base_dir = "E:\\temp\\depp_learning\\deep-learning-for-image-processing\\deep-learning-for-image-processing-master\\test_set"

    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"File '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    num_classes = len(class_indict)

    img_size = 224
    data_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    roc_data_for_plot = {}

    for time_point in time_points:
        print(f"\n--- Evaluating Bagging Ensemble for {time_point} ---")

        test_dir = os.path.join(test_data_base_dir, time_point.upper())
        model_weights_dir = os.path.join(weights_base_dir, time_point)

        assert os.path.exists(test_dir), f"Test directory '{test_dir}' not found."
        assert os.path.exists(model_weights_dir), f"Weights directory '{model_weights_dir}' not found."

        test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        all_models = []
        for i in range(1, 6):
            weight_path = os.path.join(model_weights_dir, f"fold_{i}_best_model.pth")
            assert os.path.exists(weight_path), f"Model weights '{weight_path}' not found."
            model = create_model(num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            all_models.append(model)

        all_labels = []
        all_individual_probs = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Predicting for {time_point}"):
                images, labels = images.to(device), labels.to(device)
                all_labels.extend(labels.cpu().numpy())

                batch_probs = []
                for model in all_models:
                    outputs = model(images)
                    probabilities = torch.softmax(outputs, dim=1)[:, 1]
                    batch_probs.append(probabilities.cpu().numpy())

                all_individual_probs.append(np.array(batch_probs).T)

        all_individual_probs = np.vstack(all_individual_probs)

        final_bagging_probs = np.mean(all_individual_probs, axis=1)

        fpr, tpr, _ = roc_curve(all_labels, final_bagging_probs)
        roc_auc = auc(fpr, tpr)

        roc_data_for_plot[time_point] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        print(f"✅ Bagging Ensemble for {time_point} completed. AUC: {roc_auc:.4f}")

    plt.figure(figsize=(10, 10))
    colors = ['darkorange', 'green', 'blue', 'red', 'purple']

    for i, time_point in enumerate(time_points):
        data = roc_data_for_plot[time_point]
        plt.plot(data['fpr'], data['tpr'], color=colors[i], lw=2,
                 label=f'{time_point.upper()} Bagging AUC = {data["auc"]:.4f}')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves of Bagging Ensembles for Different Time Points')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('bagging_roc_curves_by_time_point.png')
    plt.show()
    print("✅ Combined ROC curve plot saved as 'bagging_roc_curves_by_time_point.png'.")


if __name__ == '__main__':
    main()