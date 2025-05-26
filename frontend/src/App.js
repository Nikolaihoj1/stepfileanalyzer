import React, { useState, useRef } from 'react';
import { 
  Container, 
  Paper, 
  Typography, 
  Button, 
  Box, 
  CircularProgress, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  IconButton,
  TextField,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Alert
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import TuneIcon from '@mui/icons-material/Tune';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [material, setMaterial] = useState('aluminum');
  const [error, setError] = useState(null);
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [calibrationData, setCalibrationData] = useState({
    setupTime: '',
    programmingTime: '',
    machiningTime: ''
  });
  const fileInputRef = useRef();

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && (selectedFile.name.toLowerCase().endsWith('.step') || selectedFile.name.toLowerCase().endsWith('.stp'))) {
      setFile(selectedFile);
      setError(null);
      setAnalysisResults(null);
    } else if (selectedFile) {
      setError('Please select a valid STEP file (.step or .stp)');
    }
  };

  const handleMaterialChange = (event) => {
    setMaterial(event.target.value);
  };

  const handleCalibrationSubmit = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('material', material);
    formData.append('setup_time', calibrationData.setupTime);
    formData.append('programming_time', calibrationData.programmingTime);
    formData.append('machining_time', calibrationData.machiningTime);

    try {
      const response = await fetch('http://localhost:8000/calibrate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to save calibration data');
      }

      setSnackbar({
        open: true,
        message: 'Calibration data saved successfully',
        severity: 'success'
      });
      setCalibrationOpen(false);
      analyzeFile(); // Re-analyze with new calibration
    } catch (err) {
      setSnackbar({
        open: true,
        message: err.message,
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const analyzeFile = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('material', material);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error analyzing file');
      }

      const data = await response.json();
      setAnalysisResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (num === undefined || num === null) return 'N/A';
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };

  const renderAnalysisResults = () => {
    if (!analysisResults) return null;

    const { basic_info, complexity, machining_estimate, raw_stock, material_removal } = analysisResults;

    return (
      <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>Analysis Results</Typography>
        
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" color="primary">Part Information</Typography>
          <Typography>
            Volume: {formatNumber(basic_info.volume_mm3)} mm³
          </Typography>
          <Typography>
            Weight: {formatNumber(basic_info.weight_kg)} kg
          </Typography>
          <Typography>
            Dimensions (L×W×H): {basic_info.bounding_box_mm?.dimensions ? basic_info.bounding_box_mm.dimensions.map(d => formatNumber(d)).join(' × ') : 'N/A'} mm
          </Typography>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" color="primary">Raw Stock Information</Typography>
          <Typography>
            Stock Dimensions (L×W×H): {raw_stock.dimensions.map(d => formatNumber(d)).join(' × ')} mm
          </Typography>
          <Typography>
            Stock Volume: {formatNumber(raw_stock.volume_mm3)} mm³
          </Typography>
          <Typography>
            Stock Weight: {formatNumber(raw_stock.weight_kg)} kg
          </Typography>
          <Typography>
            Material to Remove: {formatNumber(material_removal.removed_volume_mm3)} mm³ ({formatNumber(material_removal.removal_percentage)}%)
          </Typography>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" color="primary">Complexity Analysis</Typography>
          <Typography>
            Face Count: {formatNumber(complexity.face_count)}
          </Typography>
          <Typography>
            Edge Count: {formatNumber(complexity.edge_count)}
          </Typography>
          <Typography>
            Vertex Count: {formatNumber(complexity.vertex_count)}
          </Typography>
          <Typography>
            Total Entities: {formatNumber(complexity.total_entities)}
          </Typography>
        </Box>

        <Box>
          <Typography variant="subtitle1" color="primary">Machining Estimate</Typography>
          <Typography>
            Complexity Score: {formatNumber(machining_estimate.complexity_score)}
          </Typography>
          <Typography>
            Estimated Machine Time: {formatNumber(machining_estimate.estimated_machine_time_minutes)} minutes
          </Typography>
          <Typography>
            Complexity Level: {machining_estimate.complexity_level}
          </Typography>
        </Box>
      </Paper>
    );
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        HMT - STEP File Analyzer 
      </Typography>

      <Paper elevation={3} sx={{ p: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={10}>
            <FormControl fullWidth>
              <InputLabel id="material-select-label">Material</InputLabel>
              <Select
                labelId="material-select-label"
                id="material-select"
                value={material}
                label="Material"
                onChange={handleMaterialChange}
              >
                <MenuItem value="aluminum">Aluminum</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={2}>
            <IconButton 
              color="primary" 
              onClick={() => setCalibrationOpen(true)}
              disabled={!file}
              title="Calibrate timing"
            >
              <TuneIcon />
            </IconButton>
          </Grid>
        </Grid>

        <Box
          sx={{
            border: '2px dashed #ccc',
            borderRadius: 2,
            p: 3,
            mt: 2,
            textAlign: 'center',
            cursor: 'pointer',
            '&:hover': {
              backgroundColor: '#f5f5f5'
            }
          }}
          onClick={() => fileInputRef.current.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".step,.stp"
            style={{ display: 'none' }}
          />
          <IconButton
            color="primary"
            aria-label="upload file"
            component="span"
            sx={{ mb: 1 }}
          >
            <CloudUploadIcon fontSize="large" />
          </IconButton>
          <Typography variant="h6" gutterBottom>
            Drop your STEP file here
          </Typography>
          <Typography variant="body2" color="textSecondary">
            or click to select
          </Typography>
          {file && (
            <Typography variant="body2" sx={{ mt: 2 }}>
              Selected: {file.name}
            </Typography>
          )}
        </Box>

        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="contained"
            onClick={analyzeFile}
            disabled={!file || loading}
            sx={{ minWidth: 200 }}
          >
            {loading ? <CircularProgress size={24} /> : 'Analyze'}
          </Button>
        </Box>

        {error && (
          <Typography color="error" sx={{ mt: 2 }}>
            Error: {error}
          </Typography>
        )}

        {renderAnalysisResults()}
      </Paper>

      <Dialog open={calibrationOpen} onClose={() => setCalibrationOpen(false)}>
        <DialogTitle>Calibrate Timing</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Enter actual timing data for this part to improve future estimates
          </Typography>
          <TextField
            fullWidth
            label="Setup Time (minutes)"
            type="number"
            value={calibrationData.setupTime}
            onChange={(e) => setCalibrationData({...calibrationData, setupTime: e.target.value})}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Programming Time (minutes)"
            type="number"
            value={calibrationData.programmingTime}
            onChange={(e) => setCalibrationData({...calibrationData, programmingTime: e.target.value})}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Machining Time (minutes)"
            type="number"
            value={calibrationData.machiningTime}
            onChange={(e) => setCalibrationData({...calibrationData, machiningTime: e.target.value})}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCalibrationOpen(false)}>Cancel</Button>
          <Button onClick={handleCalibrationSubmit} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({...snackbar, open: false})}
      >
        <Alert severity={snackbar.severity} onClose={() => setSnackbar({...snackbar, open: false})}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default App; 