from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import pickle
import os

def train_model():
    """Train the logistic regression model and return it with feature names"""
    # 1. Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target  # features and species labels
    
    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Fit logistic regression
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, iris.feature_names, iris.target_names, acc, y_test, y_pred, X_test

def save_model(model, feature_names, target_names, model_path="iris_model.pkl"):
    """Save the trained model and feature names"""
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'target_names': target_names
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {model_path}")

def load_model(model_path="iris_model.pkl"):
    """Load the trained model and feature names"""
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names'], model_data['target_names']

def predict_iris_species(features, model, feature_names, target_names):
    """Predict iris species for given features"""
    if len(features) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} features, got {len(features)}")
    
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    
    return prediction, probabilities

def main():
    parser = argparse.ArgumentParser(description='Iris Species Classification Predictor')
    parser.add_argument('--train', action='store_true', help='Train and save the model')
    parser.add_argument('--predict', nargs=4, type=float, metavar=('SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'),
                       help='Predict iris species with 4 feature values')
    parser.add_argument('--interactive', action='store_true', help='Interactive prediction mode')
    parser.add_argument('--model-path', default='iris_model.pkl', help='Path to save/load model')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training Iris Species Classification Model...")
        model, feature_names, target_names, acc, y_test, y_pred, X_test = train_model()
        
        print(f"Accuracy: {acc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Save model
        save_model(model, feature_names, target_names, args.model_path)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # Plot 2: Feature Importance (Coefficients)
        coef_importance = np.abs(model.coef_[0])  # Use first class coefficients
        sorted_idx = np.argsort(coef_importance)[::-1]

        axes[0, 1].barh(range(len(feature_names)), coef_importance[sorted_idx], color='orange')
        axes[0, 1].set_yticks(range(len(feature_names)))
        axes[0, 1].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[0, 1].set_xlabel('Absolute Coefficient Value')
        axes[0, 1].set_title('Feature Importance (Coefficient Magnitude)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Prediction Confidence Distribution
        y_pred_proba = model.predict_proba(X_test)
        max_proba = np.max(y_pred_proba, axis=1)
        axes[1, 0].hist(max_proba, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Prediction Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Prediction Confidence')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Class Distribution
        unique, counts = np.unique(y_test, return_counts=True)
        axes[1, 1].bar([target_names[i] for i in unique], counts, color=['red', 'green', 'blue'], alpha=0.7)
        axes[1, 1].set_xlabel('Species')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Test Set Class Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Iris Species Classification Analysis', fontsize=16, y=1.02)
        plt.show()

        # Additional analysis
        print("\n" + "="*50)
        print("MODEL ANALYSIS")
        print("="*50)
        print(f"Model accuracy: {acc:.1%}")
        print(f"Most important feature: {feature_names[sorted_idx[0]]}")
        print(f"Least important feature: {feature_names[sorted_idx[-1]]}")
        print(f"Average prediction confidence: {np.mean(max_proba):.3f}")
        
    elif args.predict:
        # Load model and make prediction
        model, feature_names, target_names = load_model(args.model_path)
        if model is None:
            print("Model not found. Please train the model first with --train")
            sys.exit(1)
        
        try:
            prediction, probabilities = predict_iris_species(args.predict, model, feature_names, target_names)
            predicted_species = target_names[prediction]
            confidence = probabilities[prediction]
            
            print(f"\n🌸 Iris Species Prediction")
            print("="*40)
            print(f"Predicted Species: {predicted_species}")
            print(f"Confidence: {confidence:.3f} ({confidence:.1%})")
            print(f"\nFeature Values Used:")
            for i, (name, value) in enumerate(zip(feature_names, args.predict)):
                print(f"  {name}: {value}")
            
            print(f"\nProbability Distribution:")
            for i, (species, prob) in enumerate(zip(target_names, probabilities)):
                print(f"  {species}: {prob:.3f} ({prob:.1%})")
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            sys.exit(1)
            
    elif args.interactive:
        # Interactive mode
        model, feature_names, target_names = load_model(args.model_path)
        if model is None:
            print("Model not found. Please train the model first with --train")
            sys.exit(1)
        
        print("\n🌸 Iris Species Predictor - Interactive Mode")
        print("="*60)
        print("Enter feature values for iris species prediction:")
        print("(Press Enter to use default values shown in brackets)")
        
        features = []
        default_values = [5.1, 3.5, 1.4, 0.2]  # Typical setosa values
        
        for i, name in enumerate(feature_names):
            default = default_values[i]
            try:
                value = input(f"{name} (default: {default}): ").strip()
                if value == "":
                    value = default
                else:
                    value = float(value)
                features.append(value)
            except ValueError:
                print(f"Invalid input. Using default value: {default}")
                features.append(default)
        
        try:
            prediction, probabilities = predict_iris_species(features, model, feature_names, target_names)
            predicted_species = target_names[prediction]
            confidence = probabilities[prediction]
            
            print(f"\n🌸 Prediction Result:")
            print(f"Predicted Species: {predicted_species}")
            print(f"Confidence: {confidence:.3f} ({confidence:.1%})")
            print(f"\nProbability Distribution:")
            for i, (species, prob) in enumerate(zip(target_names, probabilities)):
                print(f"  {species}: {prob:.3f} ({prob:.1%})")
        except Exception as e:
            print(f"Error making prediction: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()