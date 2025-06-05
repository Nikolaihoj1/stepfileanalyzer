"""
Test script for the HistGradientAnalyzer
This script provides a simple way to test the ML implementation
without affecting the main application.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from hist_gradient_analyzer import HistGradientAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_material(analyzer, material_type):
    """Test analyzer with a single material type"""
    try:
        logger.info(f"\nTesting material: {material_type}")
        
        # Train and evaluate
        results = analyzer.train_and_evaluate(material_type)
        
        # Print results
        logger.info("\nEvaluation Results:")
        logger.info(f"RMSE: {results['metrics']['rmse']:.2f}")
        logger.info(f"R² Score: {results['metrics']['r2_score']:.2f}")
        logger.info(f"Cross-validation RMSE: {results['metrics']['cv_rmse_mean']:.2f} ± {results['metrics']['cv_rmse_std']:.2f}")
        logger.info(f"Number of samples: {results['data_stats']['n_samples']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing {material_type}: {str(e)}")
        return None

def test_prediction(analyzer, material_type, analysis_result):
    """Test prediction with a single analysis result"""
    try:
        logger.info(f"\nTesting prediction for {material_type}")
        
        # Make prediction
        prediction = analyzer.predict_single(analysis_result)
        
        # Print results
        logger.info("\nPrediction Results:")
        logger.info(f"Predicted machining time: {prediction['predicted_time']:.2f} minutes")
        if prediction['confidence_bounds']:
            logger.info(f"Confidence bounds: [{prediction['confidence_bounds']['lower']:.2f}, "
                       f"{prediction['confidence_bounds']['upper']:.2f}]")
            
        return prediction
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None

def main():
    """Main test function"""
    try:
        # Initialize analyzer
        analyzer = HistGradientAnalyzer()
        
        # Test all materials
        materials = ['aluminum_6082', 'steel_s355', 'steel_30crni', 'stainless_316l']
        
        results = {}
        for material in materials:
            material_results = test_single_material(analyzer, material)
            if material_results:
                results[material] = material_results
                
        # Save overall results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = analyzer.results_dir / f"overall_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Error in main test function: {str(e)}")

if __name__ == "__main__":
    main()