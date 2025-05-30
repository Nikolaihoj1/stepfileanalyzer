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
  Alert,
  Tabs,
  Tab,
  TableContainer,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import TuneIcon from '@mui/icons-material/Tune';
import StepViewer from './components/StepViewer';
import HistoryView from './components/HistoryView';
import { API_BASE_URL } from './config';

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [material, setMaterial] = useState('aluminum_6082');
  const [error, setError] = useState(null);
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [calibrationData, setCalibrationData] = useState({
    setupTime: '',
    programmingTime: '',
    machiningTime: ''
  });
  const fileInputRef = useRef();
  const [geometryData, setGeometryData] = useState(null);
  const [tabValue, setTabValue] = useState(0);

  const MATERIAL_OPTIONS = [
    { value: 'aluminum_6082', label: 'Aluminum 6082', color: '#A8A8A8' },
    { value: 'steel_s355', label: 'Steel S355', color: '#6B6B6B' },
    { value: 'steel_30crni', label: 'Steel 30CrNi', color: '#4A4A4A' },
    { value: 'stainless_316l', label: 'Stainless Steel 316L', color: '#8C8C8C' }
  ];

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

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
      const response = await fetch(`${API_BASE_URL}/calibrate`, {
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
    setGeometryData(null);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('material', material);

    try {
      // First try to analyze the file
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error analyzing file');
      }

      const data = await response.json();
      setAnalysisResults(data);
      
      // Then try to get geometry data
      try {
        const geometryResponse = await fetch(`${API_BASE_URL}/geometry`, {
          method: 'POST',
          body: formData,
        });
        
        if (geometryResponse.ok) {
          const geometryData = await geometryResponse.json();
          if (geometryData.vertices && geometryData.faces) {
            setGeometryData(geometryData);
          } else {
            console.warn('Invalid geometry data structure received');
            // Don't throw error here, just continue without 3D preview
          }
        } else {
          console.warn('Failed to get geometry data');
          // Don't throw error here, just continue without 3D preview
        }
      } catch (geometryError) {
        console.warn('Error fetching geometry:', geometryError);
        // Don't throw error here, just continue without 3D preview
      }
    } catch (err) {
      setError(err.message);
      setAnalysisResults(null);
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
            Volume: {formatNumber(basic_info?.volume_mm3)} mm³
          </Typography>
          <Typography>
            Weight: {formatNumber(basic_info?.weight_kg)} kg
          </Typography>
          <Typography>
            Dimensions (L×W×H): {basic_info?.bounding_box_mm?.dimensions ? basic_info.bounding_box_mm.dimensions.map(d => formatNumber(d)).join(' × ') : 'N/A'} mm
          </Typography>
          <Typography>
            Required Machining: {machining_estimate?.required_axes}-axis
          </Typography>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" color="primary">Raw Stock Information</Typography>
          <Typography>
            Stock Dimensions (L×W×H): {raw_stock?.dimensions ? raw_stock.dimensions.map(d => formatNumber(d)).join(' × ') : 'N/A'} mm
          </Typography>
          <Typography>
            Stock Volume: {formatNumber(raw_stock?.volume_mm3)} mm³
          </Typography>
          <Typography>
            Stock Weight: {formatNumber(raw_stock?.weight_kg)} kg
          </Typography>
          <Typography>
            Material to Remove: {formatNumber(material_removal?.removed_volume_mm3)} mm³ ({formatNumber(material_removal?.removal_percentage)}%)
          </Typography>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1" color="primary">Complexity Analysis</Typography>
          <Typography>
            Face Count: {formatNumber(complexity?.face_count)}
          </Typography>
          <Typography>
            Edge Count: {formatNumber(complexity?.edge_count)}
          </Typography>
          <Typography>
            Vertex Count: {formatNumber(complexity?.vertex_count)}
          </Typography>
          <Typography>
            Total Entities: {formatNumber(complexity?.total_entities)}
          </Typography>
          <Typography>
            Required Machining: {machining_estimate?.required_axes || 3}-axis
          </Typography>
        </Box>

        <Box>
          <Typography variant="subtitle1" color="primary">Machining Estimate</Typography>
          <Typography>
            Complexity Score: {formatNumber(machining_estimate?.complexity_score)}
          </Typography>
          <Typography>
            Complexity Level: {machining_estimate?.complexity_level || 'N/A'}
          </Typography>
          <Typography>
            Base Setup Time: {formatNumber(machining_estimate?.setup_time_minutes)} minutes
          </Typography>
          <Typography>
            Base Programming Time: {formatNumber(machining_estimate?.programming_time_minutes)} minutes
          </Typography>
          <Typography>
            Base Machining Time: {formatNumber(machining_estimate?.machining_time_minutes)} minutes
          </Typography>

          {/* Batch Quantity Estimates */}
          {machining_estimate?.batch_estimates && Object.keys(machining_estimate.batch_estimates).length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" color="primary">Batch Quantity Estimates</Typography>
              <TableContainer component={Paper} sx={{ mt: 1 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Quantity</TableCell>
                      <TableCell align="right">Time per Part</TableCell>
                      <TableCell align="right">Total Time</TableCell>
                      <TableCell align="right">Setup</TableCell>
                      <TableCell align="right">Programming</TableCell>
                      <TableCell align="right">Machining/Part</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(machining_estimate.batch_estimates).map(([quantity, estimate]) => (
                      <TableRow key={quantity}>
                        <TableCell>{quantity} pcs</TableCell>
                        <TableCell align="right">{formatNumber(estimate.time_per_part)} min</TableCell>
                        <TableCell align="right">{formatNumber(estimate.total_time)} min</TableCell>
                        <TableCell align="right">{formatNumber(estimate.setup_time)} min</TableCell>
                        <TableCell align="right">{formatNumber(estimate.programming_time)} min</TableCell>
                        <TableCell align="right">{formatNumber(estimate.machining_time_per_part)} min</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}

          {machining_estimate?.similar_parts?.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">Similar Calibrated Parts:</Typography>
              {machining_estimate.similar_parts.map((part, index) => (
                <Typography key={index} variant="body2">
                  • {part.filename} (Similarity: {(part.similarity * 100).toFixed(1)}%)
                </Typography>
              ))}
            </Box>
          )}
          <Button
            variant="outlined"
            onClick={() => setCalibrationOpen(true)}
            sx={{ mt: 2 }}
          >
            Calibrate Times
          </Button>
        </Box>
      </Paper>
    );
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <img 
          src="/logo.svg" 
          alt="HMT Fast-Quote Logo" 
          style={{ 
            height: '50px',
            marginRight: '16px'
          }} 
        />
        <Typography variant="h4" component="h1">
          HMT - Fast-Quote
        </Typography>
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Analyze" />
          <Tab label="History" />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
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
                  {MATERIAL_OPTIONS.map((option) => (
                    <MenuItem 
                      key={option.value} 
                      value={option.value}
                      sx={{
                        '&:before': {
                          content: '""',
                          display: 'block',
                          width: 14,
                          height: 14,
                          mr: 1,
                          backgroundColor: option.color,
                          borderRadius: '50%'
                        }
                      }}
                    >
                      {option.label}
                    </MenuItem>
                  ))}
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

          {file && !loading && (
            <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
              <Typography variant="h6" gutterBottom>3D Preview</Typography>
              {geometryData ? (
                <StepViewer geometryData={geometryData} />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '600px' }}>
                  <Typography>No geometry data available</Typography>
                </Box>
              )}
            </Paper>
          )}

          {renderAnalysisResults()}
        </Paper>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <HistoryView material={material} />
      </TabPanel>

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