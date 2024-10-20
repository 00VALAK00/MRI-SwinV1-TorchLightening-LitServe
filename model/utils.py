import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

images_path = Path(__file__).parent.parent.joinpath('images')
cm_path = images_path / "confusion_matrix.png"


def compute_confusion_matrix(matrices, labels):
    print(matrices[0].shape)
    overall_cm = sum(matrices)

    assert overall_cm.shape == (len(labels), len(labels))

    disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm,
                                  display_labels=labels)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(str(cm_path))
    plt.axis('off')
    plt.close()

    cm = plt.imread(str(cm_path))
    cm_tensor = torch.tensor(cm).permute(2, 0, 1)  # Assuming image has shape H, W, C

    return cm_tensor
