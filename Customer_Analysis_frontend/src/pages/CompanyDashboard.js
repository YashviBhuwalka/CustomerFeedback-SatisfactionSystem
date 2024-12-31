import React, { useState } from 'react';
import axios from 'axios';

function CompanyDashboard() {
  const [file, setFile] = useState(null);
  const [customerData, setCustomerData] = useState([]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = () => {
    const formData = new FormData();
    formData.append('file', file);

    axios.post('/api/upload-data', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
      .then(response => {
        setCustomerData(response.data.customers);
        alert('File uploaded successfully');
      })
      .catch(() => alert('Error uploading file'));
  };

  const handleCheckSatisfaction = (customerId) => {
    axios.get(`/api/check-satisfaction/${customerId}`)
      .then(response => {
        alert(`Satisfaction Score for Customer ${customerId}: ${response.data.satisfactionScore}`);
      })
      .catch(() => alert('Error fetching satisfaction score'));
  };

  return (
    <div className="container">
      <h2>Company Dashboard</h2>
      <input type="file" onChange={handleFileChange} className="form-control mb-3" />
      <button onClick={handleUpload} className="btn btn-primary w-100 mb-3">Upload File</button>

      {customerData.length > 0 && (
        <div>
          <h3>Uploaded Customers</h3>
          <table className="table table-bordered">
            <thead>
              <tr>
                <th>Customer ID</th>
                <th>Age</th>
                <th>Country</th>
                <th>Gender</th>
                <th>Income</th>
                <th>Product Quality</th>
                <th>Service Quality</th>
                <th>Purchase Frequency</th>
                <th>Feedback Score</th>
                <th>Loyalty Level</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {customerData.map((customer) => (
                <tr key={customer.CustomerID}>
                  <td>{customer.CustomerID}</td>
                  <td>{customer.Age}</td>
                  <td>{customer.Country}</td>
                  <td>{customer.Gender}</td>
                  <td>{customer.Income}</td>
                  <td>{customer.ProductQuality}</td>
                  <td>{customer.ServiceQuality}</td>
                  <td>{customer.PurchaseFrequency}</td>
                  <td>{customer.FeedbackScore}</td>
                  <td>{customer.LoyaltyLevel}</td>
                  <td>
                    <button
                      className="btn btn-info"
                      onClick={() => handleCheckSatisfaction(customer.CustomerID)}
                    >
                      Check Satisfaction
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default CompanyDashboard;