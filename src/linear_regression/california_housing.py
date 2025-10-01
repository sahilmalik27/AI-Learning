from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import pickle
import os

def train_model():
    """Train the linear regression model and return it with feature names"""
    # 1. Load dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target  # y is median house value (in 100k USD)
    
    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Predict & evaluate
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, housing.feature_names, rmse, r2, y_test, y_pred

def save_model(model, feature_names, model_path="california_housing_model.pkl"):
    """Save the trained model and feature names"""
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {model_path}")

def load_model(model_path="california_housing_model.pkl"):
    """Load the trained model and feature names"""
    if not os.path.exists(model_path):
        return None, None
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names']

def predict_house_price(features, model, feature_names):
    """Predict house price for given features"""
    if len(features) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} features, got {len(features)}")
    
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    return prediction

def main():
    parser = argparse.ArgumentParser(description='California Housing Price Predictor')
    parser.add_argument('--train', action='store_true', help='Train and save the model')
    parser.add_argument('--predict', nargs=8, type=float, metavar=('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'),
                       help='Predict house price with 8 feature values')
    parser.add_argument('--interactive', action='store_true', help='Interactive prediction mode')
    parser.add_argument('--model-path', default='california_housing_model.pkl', help='Path to save/load model')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training California Housing Linear Regression Model...")
        model, feature_names, rmse, r2, y_test, y_pred = train_model()
        
        print("RMSE:", rmse)
        print("R² Score:", r2)
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        
        # Save model
        save_model(model, feature_names, args.model_path)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual House Values ($100k)')
        axes[0, 0].set_ylabel('Predicted House Values ($100k)')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values ($100k)')
        axes[0, 1].set_ylabel('Residuals ($100k)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Feature Importance (Coefficients)
        coef_importance = np.abs(model.coef_)
        sorted_idx = np.argsort(coef_importance)[::-1]

        axes[1, 0].barh(range(len(feature_names)), coef_importance[sorted_idx], color='orange')
        axes[1, 0].set_yticks(range(len(feature_names)))
        axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[1, 0].set_xlabel('Absolute Coefficient Value')
        axes[1, 0].set_title('Feature Importance (Coefficient Magnitude)')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Distribution of residuals
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residuals ($100k)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('California Housing Linear Regression Analysis', fontsize=16, y=1.02)
        plt.show()

        # Additional analysis
        print("\n" + "="*50)
        print("MODEL ANALYSIS")
        print("="*50)
        print(f"Model explains {r2:.1%} of the variance in house prices")
        print(f"Average prediction error: ${rmse*100000:.0f}")
        print(f"Most important feature: {feature_names[sorted_idx[0]]}")
        print(f"Least important feature: {feature_names[sorted_idx[-1]]}")
        
    elif args.predict:
        # Load model and make prediction
        model, feature_names = load_model(args.model_path)
        if model is None:
            print("Model not found. Please train the model first with --train")
            sys.exit(1)
        
        try:
            prediction = predict_house_price(args.predict, model, feature_names)
            print(f"\n🏠 California Housing Price Prediction")
            print("="*40)
            print(f"Predicted House Value: ${prediction*100000:.0f}")
            print(f"Predicted House Value: ${prediction:.2f} (in $100k units)")
            print("\nFeature Values Used:")
            for i, (name, value) in enumerate(zip(feature_names, args.predict)):
                print(f"  {name}: {value}")
        except Exception as e:
            print(f"Error making prediction: {e}")
            sys.exit(1)
            
    elif args.interactive:
        # Interactive mode
        model, feature_names = load_model(args.model_path)
        if model is None:
            print("Model not found. Please train the model first with --train")
            sys.exit(1)
        
        print("\n🏠 California Housing Price Predictor - Interactive Mode")
        print("="*60)
        print("Enter feature values for house price prediction:")
        print("(Press Enter to use default values shown in brackets)")
        
        features = []
        default_values = [3.5, 28.0, 5.4, 1.0, 1425.0, 3.0, 34.0, -118.0]  # Typical values
        
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
            prediction = predict_house_price(features, model, feature_names)
            print(f"\n🏠 Prediction Result:")
            print(f"Predicted House Value: ${prediction*100000:.0f}")
            print(f"Predicted House Value: ${prediction:.2f} (in $100k units)")
        except Exception as e:
            print(f"Error making prediction: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()