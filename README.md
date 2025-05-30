# HMT - Fast-Quote

A powerful STEP file analysis tool for machining time estimation and part analysis.

## Features

### Geometric Analysis
- Basic Properties:
  - Volume (mm³) and surface area (mm²)
  - Dimensions (length, width, height)
  - Bounding box coordinates
  - Weight calculation based on material

- Entity Analysis:
  - Face count
  - Edge count
  - Vertex count
  - Total entities count

- Feature Detection:
  - Holes (count, radius, location)
  - Pockets (count, area, location)
  - Small radii features
  - Required machining axes (3, 4, or 5-axis)

### Manufacturing Analysis
- Material Processing:
  - Raw stock dimensions with margins
  - Raw stock volume and weight
  - Material removal volume and percentage
  - Material-specific parameters (density, stock margin)

- Complexity Assessment:
  - Surface area to volume ratio analysis
  - Entity count scoring
  - Feature complexity evaluation
  - Overall complexity score and level classification
    - Simple
    - Moderate
    - Complex
    - Very Complex
    - Extremely Complex

- Time Estimation:
  - Setup time
  - Programming time
  - Machining time
  - Total processing time
  - Confidence scoring
  - Similar parts comparison
  - Batch processing estimates

- Machining Strategy:
  - Automatic determination of required machining axes (3, 4, or 5-axis)
  - Tool accessibility analysis
  - Feature orientation assessment
  - Undercut detection
  - Complex surface evaluation

## Tech Stack

- Backend:
  - FastAPI (Python)
  - CadQuery for STEP file processing
  - NumPy for geometric calculations
  - Trimesh for mesh analysis
  - JSON-based data persistence

- Frontend:
  - React
  - Material-UI components
  - Three.js for 3D visualization
  - Modern responsive design

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd stepfileanalyzer
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Usage

1. Upload a STEP file through the web interface
2. Select material parameters (currently supporting aluminum)
3. View detailed analysis including:
   - Geometric properties (volume, weight, dimensions)
   - Feature detection and analysis
   - Required machining axes (3, 4, or 5-axis)
   - Raw stock calculations
   - Material removal estimates
   - Complexity assessment
   - Machining time estimates
   - Batch quantity estimates

## Calibration System

The system includes a calibration feature that improves time estimates based on actual machining data:

1. Upload a STEP file
2. Enter actual machining times:
   - Setup time
   - Programming time
   - Machining time
3. The system will use this data to improve future estimates

## Analysis Methodology

The complexity analysis takes into account multiple factors:

1. Volume Complexity:
   - Based on the ratio of surface area to volume
   - Normalized and weighted for balanced scoring

2. Feature Complexity:
   - Number and types of features (holes, pockets, small radii)
   - Feature size and location analysis
   - Manufacturing difficulty assessment

3. Axis Complexity:
   - Required number of machining axes (3, 4, or 5-axis) determined by:
     - Feature orientations and accessibility
     - Undercut presence
     - Deep pocket analysis
     - Surface normal variations
     - Tool approach requirements
   - Tool accessibility evaluation
   - Setup complexity consideration

4. Precision Complexity:
   - Based on smallest radius features
   - Tolerance requirements assessment
   - Surface finish implications

## Changelog

### 2024-02-07
- Rebranded to "HMT - Fast-Quote" with new logo
- Added batch quantity estimation feature (1, 5, 10, 20, 50 pieces)
- Improved time estimation with learning curve factor for batch quantities
- Enhanced UI with detailed batch analysis table
- Fixed material removal calculation issues
- Updated calibration system to handle batch estimates

### Previous Updates
- Initial calibration system implementation
- Material parameter management
- Basic STEP file analysis features

## License

[Your chosen license]

## Contributors

[Your name/organization] 