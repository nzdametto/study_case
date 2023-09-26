from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_classifier(dataset, classifier, save=True):
    """
    Evaluate a logistic regression model on the given dataset.

    Parameters:
    - dataset: The dataset containing features and the target variable.
    - classifier: The classifier to evaluate (LogisticRegression or DummyClassifier).
    - save: Whether to save the plots as image files.
    
    Returns:
    - A dictionary with evaluation metrics including accuracy, classification report, confusion matrix, AUC score,
      and plots for confusion matrix, ROC curve, precision-recall curve, and feature importance (for Logistic Regression).
    """
    # Split data into features (X) and target (y)
    exclude_columns = ['HAS_REAL_ORDER', 'Unnamed','DEALKEY', 'DEALDETKEY', 'CHANNELDSC', 'CATEGORYDSC', 'ARTDSC', 'BRANDDSC', 'CUSTOMERID', 'CHANNELID', 'ARTID', 'BRANDID', 'CATEGORYID', 'QUOTATION_DATE', 'ARTID_DATE_CREATE', 'CUSTOMER_CREATE_DATE']
    X = dataset.drop(columns=exclude_columns)
    y = dataset['HAS_REAL_ORDER']

    # Split data into training and testing sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    
    # Fit the label encoder on your target variable and transform it
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    if classifier == 'LogisticRegression':
        # Create a pipeline with data scaling and logistic regression
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif classifier == 'DummyClassifier':
        # Create a DummyClassifier
        pipe = DummyClassifier(strategy='uniform')  # You can change the strategy here
    else:
        raise ValueError("Invalid classifier. Supported classifiers are 'LogisticRegression' and 'DummyClassifier'.")

    # Fit the model on the training data
    pipe.fit(X_train, y_train_encoded)

    # Make predictions on the testing data
    y_pred = pipe.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred)
    confusion = confusion_matrix(y_test_encoded, y_pred)
    auc_score = roc_auc_score(y_test_encoded, y_pred)

    # Create a dictionary to store the metrics
    metrics = {
        "Accuracy": accuracy,
        "Classification Report": report,
        "Confusion Matrix": confusion,
        "AUC Score": auc_score
    }

    # Create plots
    # Confusion Matrix Plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save:
        plt.savefig(f"{classifier}_confusion_matrix.png")
    plt.show()

    # ROC Curve Plot (if classifier is Logistic Regression)
    if classifier == 'LogisticRegression':
        y_pred_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test_encoded, y_pred_prob)
        roc_auc = roc_auc_score(y_test_encoded, y_pred_prob)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        if save:
            plt.savefig(f"{classifier}_roc_curve.png")
        plt.show()

    # Precision-Recall Curve Plot (if classifier is Logistic Regression)
    if classifier == 'LogisticRegression':
        precision, recall, thresholds = precision_recall_curve(y_test_encoded, y_pred_prob)
        average_precision = average_precision_score(y_test_encoded, y_pred_prob)

        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        if save:
            plt.savefig(f"{classifier}_precision_recall_curve.png")
        plt.show()

    return metrics