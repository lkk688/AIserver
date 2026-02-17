)
import matplotlib.pyplot as plt

# Optional seaborn import for enhanced visualizations
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def set_seed(seed=42):
    # Save confusion matrix plot
    cm = metrics.get('confusion_matrix')
    if cm is not None:
        if HAS_SEABORN:
            cm_path = os.path.join(task_dir, 'confusion_matrix.png')
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        else:
            # Fallback without seaborn - simple plot
            cm_path = os.path.join(task_dir, 'confusion_matrix.png')
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
