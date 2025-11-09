import React, { useState } from "react";
import { getCountryYearData } from "./services/api";
import "./CountryYearQuery.css";

export default function CountryYearQuery() {
  const [country, setCountry] = useState("");
  const [year, setYear] = useState("");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setData(null);

    // Validation
    if (!country.trim()) {
      setError("Please enter a country name");
      return;
    }

    if (!year || year < 2000 || year > 2075) {
      setError("Please enter a valid year between 2000 and 2075");
      return;
    }

    setLoading(true);

    try {
      const response = await getCountryYearData(country.trim(), parseInt(year));
      if (response.success) {
        setData(response.data);
      } else {
        setError("Failed to fetch data");
      }
    } catch (err) {
      setError(err.message || "Failed to fetch data. Please check your connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="country-year-query">
      <div className="query-container">
        <form onSubmit={handleSubmit} className="query-form">
          <div className="query-header">
            <h2>Forest Cover Query</h2>
          </div>
          
          <div className="form-inputs">
            <div className="input-group">
              <label htmlFor="country">Country</label>
              <input
                id="country"
                type="text"
                value={country}
                onChange={(e) => setCountry(e.target.value)}
                placeholder="e.g., Brazil, Canada, Bangladesh"
                disabled={loading}
              />
            </div>

            <div className="input-group">
              <label htmlFor="year">Year</label>
              <input
                id="year"
                type="number"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                placeholder="2000-2075"
                min="2000"
                max="2075"
                disabled={loading}
              />
            </div>
          </div>

          <button type="submit" className="btn-submit" disabled={loading}>
            {loading ? "Loading..." : "Query"}
          </button>

          {error && (
            <div className="error-message" role="alert">
              {error}
            </div>
          )}
        </form>

        <div className="query-results">
          <h3>Results</h3>
          {data ? (
            <>
              <div className="results-grid">
                <div className="result-item">
                  <span className="result-label">Country</span>
                  <span className="result-value">{data.country}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Year</span>
                  <span className="result-value">{data.year}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Forest Cover</span>
                  <span className="result-value">{data.forest_cover_percent.toFixed(2)}%</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Forest Area</span>
                  <span className="result-value">
                    {data.forest_area_km2.toLocaleString('en-US', {
                      maximumFractionDigits: 0
                    })} km²
                  </span>
                </div>
                <div className="result-item">
                  <span className="result-label">Total Area</span>
                  <span className="result-value">
                    {data.area_km2.toLocaleString('en-US', {
                      maximumFractionDigits: 0
                    })} km²
                  </span>
                </div>
                <div className="result-item">
                  <span className="result-label">Data Type</span>
                  <span className="result-value result-type" data-type={data.data_type}>
                    {data.data_type}
                  </span>
                </div>
              </div>
              <div className="data-type-info">
                <small>{data.data_type_description}</small>
              </div>
            </>
          ) : (
            <div className="results-placeholder">
              <div className="placeholder-icon">🌲</div>
              <p className="placeholder-text">
                Use the Forest Cover Query form to search for forest data by country and year.
              </p>
              <p className="placeholder-hint">
                Enter a country name (e.g., Brazil, Canada) and select a year between 2000-2075.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

