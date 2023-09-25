from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_logistic_regression(dataset, save=True):
    """
    Evaluate a logistic regression model on the given dataset.

    Parameters:
    - dataset: The dataset containing features and the target variable.

    Returns:
    - A dictionary with evaluation metrics including accuracy, classification report, confusion matrix, AUC score,
      and plots for confusion matrix, ROC curve, precision-recall curve, and feature importance.
    """
    # Split data into features (X) and target (y)
    exclude_columns = ['HAS_REAL_ORDER', 'Unnamed','DEALKEY', 'DEALDETKEY', 'CHANNELDSC', 'CATEGORYDSC', 'ARTDSC', 'BRANDDSC', 'CUSTOMERID', 'CHANNELID', 'ARTID', 'BRANDID', 'CATEGORYID', 'QUOTATION_DATE', 'ARTID_DATE_CREATE', 'CUSTOMER_CREATE_DATE']
    X = dataset.drop(columns=exclude_columns)
    y = dataset['HAS_REAL_ORDER']

    # Split data into training and testing sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with data scaling and logistic regression
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    # Fit the model on the training data
    pipe.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = pipe.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)

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
        plt.savefig("confusion_matrix.png") 
    plt.show()

    # ROC Curve Plot
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
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
        plt.savefig("roc_curve.png") 
    plt.show()

    # Precision-Recall Curve Plot
    precision, recall, thresholds = precision_recall_curve(y_test_encoded, y_pred_prob)
    average_precision = average_precision_score(y_test_encoded, y_pred_prob)

    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    if save:
        plt.savefig("precision_recall_curve.png")
    plt.show()

    # Feature Importance Plot
    if hasattr(pipe.named_steps['logisticregression'], 'coef_'):
        feature_importance = pipe.named_steps['logisticregression'].coef_[0]
        feature_names = X_train.columns
        sorted_idx = np.argsort(feature_importance)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        if save:
            plt.savefig("feature_importance.png")
        plt.show()

    return metrics