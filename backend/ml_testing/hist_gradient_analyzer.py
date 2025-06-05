"""
Test implementation of HistGradientBoostingRegressor for STEP file analysis.
This module works independently from the main application but uses its data.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistGradientAnalyzer:
    def __init__(self, data_dir="../data"):
        """
        Initialize the analyzer with path to existing data directory
        
        Args:
            data_dir: Path to the data directory containing material_params.json
        """
        self.data_dir = Path(data_dir)
        self.model_dir = self.data_dir / "ml_models"
        self.results_dir = self.data_dir / "ml_results"
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the model pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(
                loss='squared_error',
                learning_rate=0.1,
                max_iter=100,
                max_depth=None,
                min_samples_leaf=20,
                l2_regularization=1.0,
                max_bins=255,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-7
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
        
    def load_historical_data(self, material_type):
        """
        Load historical data from the existing material_params.json
        
        Args:
            material_type: Type of material to analyze (e.g., 'aluminum_6082')
            
        Returns:
            List of dictionaries containing historical data
        """
        try:
            material_params_file = self.data_dir / 'material_params.json'
            if not material_params_file.exists():
                raise FileNotFoundError("material_params.json not found")
                
            with open(material_params_file, 'r') as f:
                material_params = json.load(f)
                
            if material_type not in material_params:
                raise ValueError(f"Material type {material_type} not found in data")
                
            # Extract calibration data
            calibration_data = material_params[material_type].get('calibration_data', {})
            
            # Convert to list format
            historical_data = []
            for filename, data in calibration_data.items():
                if isinstance(data, dict):  # Ensure valid data format
                    historical_data.append({
                        'filename': filename,
                        'volume': data.get('volume_mm3', 0),
                        'surface_area': data.get('surface_area', 0),
                        'face_count': data.get('face_count', 0),
                        'edge_count': data.get('edge_count', 0),
                        'vertex_count': data.get('vertex_count', 0),
                        'sa_to_vol_ratio': data.get('sa_to_vol_ratio', 0),
                        'removal_percentage': data.get('material_removal', {}).get('removal_percentage', 0),
                        'total_entities': data.get('total_entities', 0),
                        'actual_machining_time': data.get('machining_time', 0)
                    })
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            raise
            
    def prepare_features(self, data):
        """
        Prepare feature matrix from historical data
        """
        try:
            X = np.array([[
                d['volume'],
                d['surface_area'],
                d['face_count'],
                d['edge_count'],
                d['vertex_count'],
                d['sa_to_vol_ratio'],
                d['removal_percentage'],
                d['total_entities'],
                d['face_count'] / d['volume'] if d['volume'] > 0 else 0,  # face_density
                d['edge_count'] / d['volume'] if d['volume'] > 0 else 0   # edge_density
            ] for d in data])
            
            y = np.array([d['actual_machining_time'] for d in data])
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
            
    def train_and_evaluate(self, material_type):
        """
        Train the model and evaluate its performance
        
        Args:
            material_type: Type of material to analyze
            
        Returns:
            Dictionary containing evaluation metrics and results
        """
        try:
            # Load and prepare data
            historical_data = self.load_historical_data(material_type)
            if len(historical_data) < 5:
                raise ValueError(f"Insufficient data for material {material_type}. Need at least 5 samples.")
                
            X, y = self.prepare_features(historical_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.pipeline.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            r2 = self.pipeline.score(X_test, y_test)
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                self.pipeline, X, y, 
                cv=5, 
                scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Generate plots
            self._generate_evaluation_plots(y_test, y_pred, material_type)
            
            # Save results
            results = {
                'material_type': material_type,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'rmse': rmse,
                    'r2_score': r2,
                    'cv_rmse_mean': cv_rmse.mean(),
                    'cv_rmse_std': cv_rmse.std()
                },
                'data_stats': {
                    'n_samples': len(X),
                    'n_features': len(self.feature_names)
                }
            }
            
            # Save results to file
            results_file = self.results_dir / f"results_{material_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise
            
    def _generate_evaluation_plots(self, y_true, y_pred, material_type):
        """Generate evaluation plots"""
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot of predicted vs actual values
            ax1.scatter(y_true, y_pred, alpha=0.5)
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Machining Time')
            ax1.set_ylabel('Predicted Machining Time')
            ax1.set_title('Predicted vs Actual Values')
            
            # Residuals plot
            residuals = y_pred - y_true
            ax2.scatter(y_pred, residuals, alpha=0.5)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Machining Time')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals Plot')
            
            # Save plot
            plot_file = self.results_dir / f"evaluation_plots_{material_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            
    def predict_single(self, analysis_result):
        """
        Make prediction for a single analysis result
        
        Args:
            analysis_result: Dictionary containing analysis results from main application
            
        Returns:
            Dictionary containing prediction and confidence bounds
        """
        try:
            # Extract features
            features = np.array([[
                analysis_result['basic_info']['volume_mm3'],
                analysis_result['complexity']['surface_area_mm2'],
                analysis_result['complexity']['face_count'],
                analysis_result['complexity']['edge_count'],
                analysis_result['complexity']['vertex_count'],
                analysis_result['complexity']['surface_area_to_volume_ratio'],
                analysis_result['material_removal']['removal_percentage'],
                analysis_result['complexity']['total_entities'],
                analysis_result['complexity']['face_count'] / analysis_result['basic_info']['volume_mm3'],
                analysis_result['complexity']['edge_count'] / analysis_result['basic_info']['volume_mm3']
            ]])
            
            # Make prediction
            prediction = self.pipeline.predict(features)[0]
            
            # Get regressor from pipeline
            regressor = self.pipeline.named_steps['regressor']
            
            # Try to calculate confidence bounds
            confidence_bounds = None
            if hasattr(regressor, 'staged_predict'):
                staged_preds = np.array(list(regressor.staged_predict(features)))
                confidence_bounds = {
                    'lower': float(prediction - 2 * staged_preds.std()),
                    'upper': float(prediction + 2 * staged_preds.std())
                }
            
            return {
                'predicted_time': float(prediction),
                'confidence_bounds': confidence_bounds
            }
            
        except Exception as e:
            logger.error(f"Error in predict_single: {str(e)}")
            raise

def main():
    """
    Main function for testing the HistGradientAnalyzer
    """
    try:
        # Initialize analyzer
        analyzer = HistGradientAnalyzer()
        
        # Test with each material type
        materials = ['aluminum_6082', 'steel_s355', 'steel_30crni', 'stainless_316l']
        
        for material in materials:
            try:
                logger.info(f"\nTesting material: {material}")
                results = analyzer.train_and_evaluate(material)
                logger.info(f"Results for {material}:")
                logger.info(json.dumps(results, indent=2))
            except ValueError as ve:
                logger.warning(f"Skipping {material}: {str(ve)}")
            except Exception as e:
                logger.error(f"Error processing {material}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()