// Determine the base URL for the API
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE_URL = isLocalhost 
    ? 'http://localhost:8000'
    : `http://${window.location.hostname}:8000`;

export { API_BASE_URL }; 