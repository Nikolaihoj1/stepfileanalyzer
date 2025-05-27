# HMT - Fast-Quote

A powerful STEP file analysis tool for machining time estimation and part analysis.

## Features

- STEP file geometry analysis
- Accurate volume and weight calculations
- Raw stock calculation with margins
- Material removal estimation
- Machining time prediction with calibration system
- Setup and programming time estimation
- Material parameter management
- Persistent calibration data storage
- Batch quantity estimation (1, 5, 10, 20, 50 pieces)

## Tech Stack

- Backend:
  - FastAPI (Python)
  - NumPy for geometric calculations
  - STEP file parsing with regex
  - JSON-based data persistence

- Frontend:
  - React
  - Material-UI components
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
   - Part volume and weight
   - Raw stock dimensions
   - Material removal calculations
   - Estimated machining times
   - Complexity analysis
   - Batch quantity estimates

## Calibration System

The system includes a calibration feature that improves time estimates based on actual machining data:

1. Upload a STEP file
2. Enter actual machining times:
   - Setup time
   - Programming time
   - Machining time
3. The system will use this data to improve future estimates

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