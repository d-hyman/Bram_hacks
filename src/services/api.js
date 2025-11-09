// API service for backend calls
const API_BASE_URL = 'http://localhost:8000';

export async function getCountryYearData(country, year) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/predictions/country/${encodeURIComponent(country)}/year/${year}`
    );
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching country/year data:', error);
    throw error;
  }
}

