# Machine Learning Implementation Guide for STEP File Analyzer

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Implementation Steps](#implementation-steps)
4. [Code Examples](#code-examples)
5. [Integration Guide](#integration-guide)
6. [Best Practices](#best-practices)

## Overview

This guide outlines how to implement machine learning capabilities into the STEP file analyzer application using scikit-learn.

### Key Features to Implement
- Machining time prediction
- Part complexity classification
- Feature importance analysis
- Similarity search improvements

## Prerequisites

```bash
# Required Python packages
pip install scikit-learn numpy pandas
```

### Data Requirements
- Historical machining times
- Part analysis results
- Calibration data
- Material properties

## Implementation Steps

### 1. Basic Time Prediction Model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class MachiningTimePredictor:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        
    def prepare_features(self, analysis_result):
        """Extract features from analysis results"""
        return {
            'volume': analysis_result['basic_info']['volume_mm3'],
            'surface_area': analysis_result['complexity']['surface_area_mm2'],
            'face_count': analysis_result['complexity']['face_count'],
            'edge_count': analysis_result['complexity']['edge_count'],
            'vertex_count': analysis_result['complexity']['vertex_count'],
            'sa_to_vol_ratio': analysis_result['complexity']['surface_area_to_volume_ratio'],
            'removal_percentage': analysis_result['material_removal']['removal_percentage'],
            'total_entities': analysis_result['complexity']['total_entities']
        }
    
    def train(self, historical_data):
        """Train the model using historical data"""
        X = np.array([[
            data['volume'],
            data['surface_area'],
            data['face_count'],
            data['edge_count'],
            data['vertex_count'],
            data['sa_to_vol_ratio'],
            data['removal_percentage'],
            data['total_entities']
        ] for data in historical_data])
        
        y = np.array([data['actual_machining_time'] for data in historical_data])
        
        self.pipeline.fit(X, y)
        
        scores = cross_val_score(self.pipeline, X, y, cv=5)
        return {
            'mean_cv_score': scores.mean(),
            'cv_score_std': scores.std()
        }
    
    def predict(self, analysis_result):
        """Predict machining time for new part"""
        features = self.prepare_features(analysis_result)
        X = np.array([[
            features['volume'],
            features['surface_area'],
            features['face_count'],
            features['edge_count'],
            features['vertex_count'],
            features['sa_to_vol_ratio'],
            features['removal_percentage'],
            features['total_entities']
        ]])
        
        return self.pipeline.predict(X)[0]
```

### 1.1 Advanced Time Prediction with HistGradientBoostingRegressor

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

class AdvancedTimePredictor:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(
                loss='squared_error',  # for regression tasks
                learning_rate=0.1,
                max_iter=100,          # number of boosting stages
                max_depth=None,        # auto-determines depth
                min_samples_leaf=20,   # minimum samples per leaf
                l2_regularization=1.0, # L2 regularization
                max_bins=255,          # number of bins for numerical features
                warm_start=False,      # whether to reuse previous solution
                early_stopping=True,   # use early stopping
                validation_fraction=0.1,# fraction of data for early stopping
                n_iter_no_change=10,   # early stopping patience
                tol=1e-7              # tolerance for early stopping
            ))
        ])
        
        self.feature_names = [
            'volume',
            'surface_area',
            'face_count',
            'edge_count',
            'vertex_count',
            'sa_to_vol_ratio',
            'removal_percentage',
            'total_entities',
            'face_density',
            'edge_density'
        ]
    
    def prepare_features(self, analysis_result):
        """Extract and compute advanced features from analysis results"""
        basic_features = {
            'volume': analysis_result['basic_info']['volume_mm3'],
            'surface_area': analysis_result['complexity']['surface_area_mm2'],
            'face_count': analysis_result['complexity']['face_count'],
            'edge_count': analysis_result['complexity']['edge_count'],
            'vertex_count': analysis_result['complexity']['vertex_count'],
            'sa_to_vol_ratio': analysis_result['complexity']['surface_area_to_volume_ratio'],
            'removal_percentage': analysis_result['material_removal']['removal_percentage'],
            'total_entities': analysis_result['complexity']['total_entities']
        }
        
        # Compute additional derived features
        volume = basic_features['volume']
        basic_features['face_density'] = basic_features['face_count'] / volume if volume > 0 else 0
        basic_features['edge_density'] = basic_features['edge_count'] / volume if volume > 0 else 0
        
        return basic_features
    
    def train(self, historical_data):
        """Train the model using historical data with cross-validation"""
        try:
            # Prepare feature matrix
            X = np.array([[
                data['volume'],
                data['surface_area'],
                data['face_count'],
                data['edge_count'],
                data['vertex_count'],
                data['sa_to_vol_ratio'],
                data['removal_percentage'],
                data['total_entities'],
                data['face_count'] / data['volume'] if data['volume'] > 0 else 0,
                data['edge_count'] / data['volume'] if data['volume'] > 0 else 0
            ] for data in historical_data])
            
            # Prepare target values
            y = np.array([data['actual_machining_time'] for data in historical_data])
            
            # Train the pipeline
            self.pipeline.fit(X, y)
            
            # Get feature importances (if available)
            regressor = self.pipeline.named_steps['regressor']
            importances = None
            if hasattr(regressor, 'feature_importances_'):
                importances = dict(zip(self.feature_names, regressor.feature_importances_))
            
            # Calculate cross-validation scores
            scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)  # Convert MSE to RMSE
            
            return {
                'mean_rmse': rmse_scores.mean(),
                'rmse_std': rmse_scores.std(),
                'feature_importances': importances,
                'n_samples_trained': len(X)
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, analysis_result):
        """Make predictions with uncertainty estimates"""
        try:
            features = self.prepare_features(analysis_result)
            X = np.array([[
                features['volume'],
                features['surface_area'],
                features['face_count'],
                features['edge_count'],
                features['vertex_count'],
                features['sa_to_vol_ratio'],
                features['removal_percentage'],
                features['total_entities'],
                features['face_density'],
                features['edge_density']
            ]])
            
            # Make prediction
            prediction = self.pipeline.predict(X)[0]
            
            # Calculate confidence bounds (if possible)
            confidence_bounds = None
            regressor = self.pipeline.named_steps['regressor']
            if hasattr(regressor, 'staged_predict'):
                staged_preds = np.array(list(regressor.staged_predict(X)))
                confidence_bounds = {
                    'lower': prediction - 2 * staged_preds.std(),
                    'upper': prediction + 2 * staged_preds.std()
                }
            
            return {
                'predicted_time': prediction,
                'confidence_bounds': confidence_bounds,
                'features_used': features
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

# Example usage in FastAPI endpoint
@app.post("/analyze-ml-advanced")
async def analyze_with_advanced_ml(
    file: UploadFile = File(...),
    material: str = Form(...)
):
    """Enhanced analysis endpoint with advanced ML predictions"""
    try:
        # Get basic analysis results
        basic_analysis = await analyze_step_file(file, material)
        
        # Make prediction with advanced model
        advanced_predictor = AdvancedTimePredictor()
        prediction_result = advanced_predictor.predict(basic_analysis)
        
        return {
            **basic_analysis,
            'ml_predictions': {
                'advanced_time_prediction': prediction_result
            }
        }
        
    except Exception as e:
        logger.error(f"Advanced ML analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Advanced ML analysis failed: {str(e)}"
        )

### 2. Complexity Classification

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

class ComplexityClassifier:
    def __init__(self):
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        self.label_encoder = LabelEncoder()
        
    def prepare_features(self, analysis_result):
        """Extract features for complexity classification"""
        return {
            'sa_to_vol_ratio': analysis_result['complexity']['surface_area_to_volume_ratio'],
            'total_entities': analysis_result['complexity']['total_entities'],
            'removal_percentage': analysis_result['material_removal']['removal_percentage'],
            'face_density': (analysis_result['complexity']['face_count'] / 
                           analysis_result['basic_info']['volume_mm3'])
        }
    
    def train(self, historical_data):
        """Train the complexity classifier"""
        X = np.array([[
            data['sa_to_vol_ratio'],
            data['total_entities'],
            data['removal_percentage'],
            data['face_density']
        ] for data in historical_data])
        
        y = self.label_encoder.fit_transform([data['complexity_level'] for data in historical_data])
        
        self.classifier.fit(X, y)
        
        return dict(zip(
            ['sa_to_vol_ratio', 'total_entities', 'removal_percentage', 'face_density'],
            self.classifier.feature_importances_
        ))
    
    def predict(self, analysis_result):
        """Predict complexity level for new part"""
        features = self.prepare_features(analysis_result)
        X = np.array([[
            features['sa_to_vol_ratio'],
            features['total_entities'],
            features['removal_percentage'],
            features['face_density']
        ]])
        
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        return {
            'complexity_level': self.label_encoder.inverse_transform([prediction])[0],
            'confidence': max(probabilities),
            'class_probabilities': dict(zip(
                self.label_encoder.classes_,
                probabilities
            ))
        }
```

### 3. FastAPI Integration

```python
from fastapi import FastAPI, File, UploadFile, Form
from typing import Dict

# Initialize models
time_predictor = MachiningTimePredictor()
complexity_classifier = ComplexityClassifier()

@app.post("/analyze-ml")
async def analyze_with_ml(file: UploadFile = File(...), material: str = Form(...)):
    """Enhanced analysis endpoint with ML predictions"""
    try:
        # Get basic analysis results
        basic_analysis = await analyze_step_file(file, material)
        
        # Add ML-based predictions
        ml_predictions = {
            'predicted_machining_time': time_predictor.predict(basic_analysis),
            'complexity_analysis': complexity_classifier.predict(basic_analysis)
        }
        
        return {
            **basic_analysis,
            'ml_predictions': ml_predictions
        }
        
    except Exception as e:
        logger.error(f"ML analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML analysis failed: {str(e)}")

@app.post("/calibrate-ml")
async def calibrate_ml_models(
    calibration_data: Dict,
    material: str = Form(...)
):
    """Endpoint to retrain ML models"""
    try:
        time_training_results = time_predictor.train(calibration_data['historical_data'])
        complexity_training_results = complexity_classifier.train(calibration_data['historical_data'])
        
        return {
            'time_predictor_metrics': time_training_results,
            'complexity_feature_importance': complexity_training_results
        }
        
    except Exception as e:
        logger.error(f"ML calibration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML calibration failed: {str(e)}")
```

### 4. Feature Importance Analysis

```python
from sklearn.inspection import permutation_importance

class FeatureAnalyzer:
    def analyze_feature_importance(self, model, X, y):
        """Analyze feature importance"""
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42
        )
        
        importance_scores = result.importances_mean
        
        feature_importance = dict(zip(
            self.feature_names,
            importance_scores
        ))
        
        return sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def generate_feature_report(self, analysis_result):
        """Generate feature analysis report"""
        features = self.prepare_features(analysis_result)
        importance = self.analyze_feature_importance(
            self.model,
            np.array([list(features.values())]),
            None
        )
        
        return {
            'feature_importance': importance,
            'feature_statistics': {
                name: {
                    'value': value,
                    'percentile': self.calculate_percentile(name, value)
                }
                for name, value in features.items()
            }
        }
```

## Integration Guide

### 1. Model Storage
- Save trained models using joblib
- Store in a dedicated models directory
- Implement version control for models

```python
import joblib

# Save model
joblib.dump(model, 'models/time_predictor_v1.joblib')

# Load model
model = joblib.load('models/time_predictor_v1.joblib')
```

### 2. Data Management
- Create a database table for training data
- Implement data versioning
- Regular model retraining schedule

### 3. Error Handling
- Implement robust error handling
- Log prediction errors
- Monitor model performance

## Best Practices

1. Data Quality
- Validate input data
- Handle missing values
- Scale features appropriately
- Regular data cleaning

2. Model Maintenance
- Monitor model performance
- Retrain periodically
- Version control models
- Track prediction accuracy

3. Testing
- Unit tests for ML components
- Integration tests
- Performance benchmarks
- Cross-validation

4. Production Deployment
- Model versioning
- Fallback mechanisms
- Performance monitoring
- Error logging

## Next Steps

1. Start with basic time prediction
2. Collect and validate training data
3. Implement simple model first
4. Add complexity gradually
5. Monitor and improve

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python Machine Learning Guide](https://scikit-learn.org/stable/tutorial/index.html) 