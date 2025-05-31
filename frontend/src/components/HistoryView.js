import React, { useState, useEffect } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Collapse,
  Box,
  Typography,
  Chip,
  Grid,
  Card,
  CardContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack
} from '@mui/material';
import { KeyboardArrowDown, KeyboardArrowUp, Search } from '@mui/icons-material';
import { API_BASE_URL } from '../config';

// Format date to YYYY-MM-DD
const formatDate = (dateString) => {
  return new Date(dateString).toLocaleDateString('en-GB', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  });
};

function StatCard({ title, value, subtitle }) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography color="textSecondary" gutterBottom>
          {title}
        </Typography>
        <Typography variant="h4" component="div">
          {value}
        </Typography>
        {subtitle && (
          <Typography color="textSecondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}

function Row({ row, materialName }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <TableRow 
        sx={{ 
          '& > *': { borderBottom: 'unset' },
          backgroundColor: row.is_calibrated ? 'rgba(25, 118, 210, 0.04)' : 'inherit'
        }}
      >
        <TableCell>
          <IconButton
            aria-label="expand row"
            size="small"
            onClick={() => setOpen(!open)}
          >
            {open ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
          </IconButton>
        </TableCell>
        <TableCell component="th" scope="row">
          {row.filename}
          {row.is_calibrated && (
            <Chip
              label="Calibrated"
              color="primary"
              size="small"
              sx={{ ml: 1 }}
            />
          )}
        </TableCell>
        <TableCell>
          <Chip
            label={materialName}
            color="default"
            size="small"
            sx={{ backgroundColor: 'rgba(0, 0, 0, 0.08)' }}
          />
        </TableCell>
        <TableCell align="right">{(row.volume_mm3 / 1000).toFixed(2)} cm³</TableCell>
        <TableCell align="right">{row.complexity_score.toFixed(2)}</TableCell>
        <TableCell align="right">
          {row.material_removal ? `${row.material_removal.removal_percentage.toFixed(2)}%` : 'N/A'}
        </TableCell>
        <TableCell align="right">{formatDate(row.timestamp)}</TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={7}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ margin: 1 }}>
              <Typography variant="h6" gutterBottom component="div">
                Details
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Material Properties
                    </Typography>
                    <Stack direction="row" spacing={2}>
                      <Typography variant="body2">
                        <strong>Material:</strong> {materialName}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Density:</strong> {row.material_density || 'N/A'} g/cm³
                      </Typography>
                      <Typography variant="body2">
                        <strong>Machinability Rating:</strong> {row.machinability_rating || 'N/A'}
                      </Typography>
                    </Stack>
                  </Box>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    {row.is_calibrated ? 'Calibrated Times' : 'Estimated Times'}
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Setup Time</TableCell>
                        <TableCell>Programming Time</TableCell>
                        <TableCell>Machining Time</TableCell>
                        <TableCell>Total Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>
                          {row.is_calibrated ? (
                            <Typography variant="body2" color="primary">
                              {row.setup_time} min (calibrated)
                            </Typography>
                          ) : (
                            `${row.machining_estimate?.setup_time_minutes || 'N/A'} min`
                          )}
                        </TableCell>
                        <TableCell>
                          {row.is_calibrated ? (
                            <Typography variant="body2" color="primary">
                              {row.programming_time} min (calibrated)
                            </Typography>
                          ) : (
                            `${row.machining_estimate?.programming_time_minutes || 'N/A'} min`
                          )}
                        </TableCell>
                        <TableCell>
                          {row.is_calibrated ? (
                            <Typography variant="body2" color="primary">
                              {row.machining_time} min (calibrated)
                            </Typography>
                          ) : (
                            `${row.machining_estimate?.machining_time_minutes || 'N/A'} min`
                          )}
                        </TableCell>
                        <TableCell>
                          {row.is_calibrated ? (
                            <Typography variant="body2" color="primary">
                              {row.setup_time + row.programming_time + row.machining_time} min (calibrated)
                            </Typography>
                          ) : (
                            `${row.machining_estimate?.estimated_machine_time_minutes || 'N/A'} min`
                          )}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </Grid>
                {(row.is_calibrated || row.machining_estimate?.batch_estimates) && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom>
                      Batch Estimates
                      {row.is_calibrated && (
                        <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                          (Based on calibrated times)
                        </Typography>
                      )}
                    </Typography>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Quantity</TableCell>
                          <TableCell>Time per Part</TableCell>
                          <TableCell>Total Time</TableCell>
                          <TableCell>Setup</TableCell>
                          <TableCell>Programming</TableCell>
                          <TableCell>Machining/Part</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(row.machining_estimate?.batch_estimates || {}).map(([qty, data]) => (
                          <TableRow key={qty}>
                            <TableCell>{qty} pcs</TableCell>
                            <TableCell>{data.time_per_part.toFixed(2)} min</TableCell>
                            <TableCell>{data.total_time.toFixed(2)} min</TableCell>
                            <TableCell>{data.setup_time.toFixed(2)} min</TableCell>
                            <TableCell>{data.programming_time.toFixed(2)} min</TableCell>
                            <TableCell>{data.machining_time_per_part.toFixed(2)} min</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Grid>
                )}
              </Grid>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

function HistoryView({ material }) {
  const [historyData, setHistoryData] = useState({
    history: [],
    total_entries: 0,
    calibrated_count: 0,
    analyzed_count: 0
  });
  const [materialInfo, setMaterialInfo] = useState(null);
  
  // Filtering states
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');

  useEffect(() => {
    // Fetch material info
    fetch(`${API_BASE_URL}/materials`)
      .then(response => response.json())
      .then(data => {
        setMaterialInfo(data[material]);
      })
      .catch(error => {
        console.error('Error fetching material info:', error);
      });

    // Fetch history
    fetch(`${API_BASE_URL}/history?material=${material}`)
      .then(response => response.json())
      .then(data => {
        setHistoryData(data);
      })
      .catch(error => {
        console.error('Error fetching history:', error);
      });
  }, [material]);

  // Filter and search logic
  const filteredHistory = historyData.history.filter(item => {
    const matchesSearch = item.filename.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === 'all' || 
      (filterStatus === 'calibrated' && item.is_calibrated) ||
      (filterStatus === 'analyzed' && !item.is_calibrated);
    return matchesSearch && matchesFilter;
  });

  return (
    <Box sx={{ width: '100%' }}>
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <StatCard 
            title="Total Parts" 
            value={historyData.total_entries}
            subtitle="Analyzed + Calibrated"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard 
            title="Calibrated Parts" 
            value={historyData.calibrated_count}
            subtitle="With actual machining times"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard 
            title="Analyzed Parts" 
            value={historyData.analyzed_count}
            subtitle="Pending calibration"
          />
        </Grid>
      </Grid>

      {/* Search and Filter Controls */}
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 3 }}>
        <TextField
          label="Search by filename"
          variant="outlined"
          size="small"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ minWidth: 200 }}
          InputProps={{
            startAdornment: <Search color="action" sx={{ mr: 1 }} />
          }}
        />
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Filter Status</InputLabel>
          <Select
            value={filterStatus}
            label="Filter Status"
            onChange={(e) => setFilterStatus(e.target.value)}
          >
            <MenuItem value="all">All Parts</MenuItem>
            <MenuItem value="calibrated">Calibrated Only</MenuItem>
            <MenuItem value="analyzed">Analyzed Only</MenuItem>
          </Select>
        </FormControl>
      </Stack>
      
      <TableContainer component={Paper}>
        <Table aria-label="collapsible table">
          <TableHead>
            <TableRow>
              <TableCell />
              <TableCell>Filename</TableCell>
              <TableCell>Material</TableCell>
              <TableCell align="right">Volume</TableCell>
              <TableCell align="right">Complexity Score</TableCell>
              <TableCell align="right">Material Removal</TableCell>
              <TableCell align="right">Date</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredHistory.map((row, index) => (
              <Row 
                key={`${row.filename}-${index}`} 
                row={row} 
                materialName={materialInfo?.name || material}
              />
            ))}
            {filteredHistory.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body1" sx={{ py: 2 }}>
                    No matching parts found
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default HistoryView;