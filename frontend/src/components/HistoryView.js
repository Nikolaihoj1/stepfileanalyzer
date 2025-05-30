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
  Chip
} from '@mui/material';
import { KeyboardArrowDown, KeyboardArrowUp } from '@mui/icons-material';
import { API_BASE_URL } from '../config';

function Row({ row }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <TableRow sx={{ '& > *': { borderBottom: 'unset' } }}>
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
        <TableCell align="right">{(row.volume_mm3 / 1000).toFixed(2)} cm³</TableCell>
        <TableCell align="right">{row.complexity_score.toFixed(2)}</TableCell>
        <TableCell align="right">
          {row.material_removal ? `${row.material_removal.removal_percentage.toFixed(2)}%` : 'N/A'}
        </TableCell>
        <TableCell align="right">{new Date(row.timestamp).toLocaleString()}</TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ margin: 1 }}>
              <Typography variant="h6" gutterBottom component="div">
                Timing Details
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
                    <TableCell>{row.setup_time} min</TableCell>
                    <TableCell>{row.programming_time} min</TableCell>
                    <TableCell>{row.machining_time} min</TableCell>
                    <TableCell>
                      {(row.setup_time + row.programming_time + row.machining_time)} min
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

function HistoryView({ material }) {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/history?material=${material}`)
      .then(response => response.json())
      .then(data => {
        setHistory(data.history);
      })
      .catch(error => {
        console.error('Error fetching history:', error);
      });
  }, [material]);

  return (
    <TableContainer component={Paper}>
      <Table aria-label="collapsible table">
        <TableHead>
          <TableRow>
            <TableCell />
            <TableCell>Filename</TableCell>
            <TableCell align="right">Volume</TableCell>
            <TableCell align="right">Complexity Score</TableCell>
            <TableCell align="right">Material Removal</TableCell>
            <TableCell align="right">Date</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {history.map((row, index) => (
            <Row key={`${row.filename}-${index}`} row={row} />
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default HistoryView; 