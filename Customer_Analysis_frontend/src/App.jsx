
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import CustomerLogin from './pages/CustomerLogin';
import CompanyLogin from './pages/CompanyLogin';
import CustomerFeedback from './pages/CustomerFeedback';
import CompanyDashboard from './pages/CompanyDashboard';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<CustomerLogin />} />
        <Route path="/company-login" element={<CompanyLogin />} />
        <Route path="/feedback" element={<CustomerFeedback />} />
        <Route path="/company-dashboard" element={<CompanyDashboard />} />
      </Routes>
    </Router>
  );
}

export default App;