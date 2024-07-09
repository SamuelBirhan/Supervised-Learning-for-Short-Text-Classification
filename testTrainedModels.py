import argparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", default="test_dataset", help="Path to the test dataset")
    args = parser.parse_args()

    # Define the paths to the saved models
    model_paths = [
        'trained_model_Count_Vectorizer_Naive_Bayes.pkl',
        'trained_model_TF-IDF_Naive_Bayes.pkl',
        'trained_model_TF-IDF_SVM.pkl'
    ]

    # Load the test data
    test_data = load_files(args.data, encoding="latin-1")

    # List of model names for plotting purposes
    model_names = [
        'Count Vectorizer -> Naive Bayes',
        'TF-IDF -> Naive Bayes',
        'TF-IDF -> SVM'
    ]

    # Iterate through each model path and evaluate performance
    for model_path, model_name in zip(model_paths, model_names):
        # Load the trained model
        model = joblib.load(model_path)

        # Make predictions on the test data
        predictions = model.predict(test_data.data)

        # Calculate accuracy
        accuracy = accuracy_score(test_data.target, predictions)
        print(f"Accuracy for {model_name}: {accuracy}")

        # Generate classification report
        print(f"Classification Report for {model_name}:")
        print(classification_report(test_data.target, predictions, target_names=test_data.target_names))

        # Generate confusion matrix
        cm = confusion_matrix(test_data.target, predictions)

        # Plot confusion matrix
        plot_confusion_matrix(cm, labels=test_data.target_names, title=f"Confusion Matrix for {model_name}")
