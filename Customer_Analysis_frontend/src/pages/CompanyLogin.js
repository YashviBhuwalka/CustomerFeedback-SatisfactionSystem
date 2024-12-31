import React from 'react';
import { useNavigate } from 'react-router-dom';
import './styles.css';

function CompanyLogin() {
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    navigate('/company-dashboard');
  };

  return (
    <div className="container login-container">
      <h2>Company Login</h2>
      <form onSubmit={handleSubmit}>
        <input type="email" placeholder="Email" required className="form-control mb-3" />
        <input type="password" placeholder="Password" required className="form-control mb-3" />
        <button type="submit" className="btn btn-primary w-100">Login</button>
      </form>
    </div>
  );
}

export default CompanyLogin;