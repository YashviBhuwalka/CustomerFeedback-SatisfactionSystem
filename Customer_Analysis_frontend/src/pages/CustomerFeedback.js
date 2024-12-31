import React from 'react';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import axios from 'axios';

function CustomerFeedback() {
  const formik = useFormik({
    initialValues: {
      customerID: '',
      age: '',
      country: '',
      gender: '',
      income: '',
      productQuality: '',
      serviceQuality: '',
      purchaseFrequency: '',
      feedbackScore: '',
      loyaltyLevel: '',
    },
    validationSchema: Yup.object({
      customerID: Yup.number().required('Customer ID is required'),
      age: Yup.number().min(0).required('Age is required'),
      country: Yup.string().required('Country is required'),
      gender: Yup.string().required('Gender is required'),
      income: Yup.number().min(0).required('Income is required'),
      productQuality: Yup.number().min(1).max(5).required('Product quality rating is required'),
      serviceQuality: Yup.number().min(1).max(5).required('Service quality rating is required'),
      purchaseFrequency: Yup.string().required('Purchase frequency is required'),
      feedbackScore: Yup.string().required('Feedback score is required'),
      loyaltyLevel: Yup.string().required('Loyalty level is required'),
    }),
    onSubmit: (values) => {
      axios.post('/api/submit-feedback', values)
        .then(() => alert('Feedback submitted successfully'))
        .catch(() => alert('Error submitting feedback'));
    },
  });

  return (
    <div className="container">
      <h2>Customer Feedback</h2>
      <form onSubmit={formik.handleSubmit}>
        <input
          type="text"
          placeholder="Customer ID"
          {...formik.getFieldProps('customerID')}
          className="form-control mb-3"
        />
        <input
          type="number"
          placeholder="Age"
          {...formik.getFieldProps('age')}
          className="form-control mb-3"
        />
        <input
          type="text"
          placeholder="Country"
          {...formik.getFieldProps('country')}
          className="form-control mb-3"
        />
        <select {...formik.getFieldProps('gender')} className="form-control mb-3">
          <option value="">Select Gender</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
          <option value="Other">Other</option>
        </select>
        <input
          type="number"
          placeholder="Income"
          {...formik.getFieldProps('income')}
          className="form-control mb-3"
        />
        <select {...formik.getFieldProps('productQuality')} className="form-control mb-3">
          <option value="">Rate Product Quality</option>
          {[1, 2, 3, 4, 5].map((val) => (
            <option value={val} key={val}>{val}</option>
          ))}
        </select>
        <select {...formik.getFieldProps('serviceQuality')} className="form-control mb-3">
          <option value="">Rate Service Quality</option>
          {[1, 2, 3, 4, 5].map((val) => (
            <option value={val} key={val}>{val}</option>
          ))}
        </select>
        <select {...formik.getFieldProps('purchaseFrequency')} className="form-control mb-3">
          <option value="">Select Purchase Frequency</option>
          {['Weekly', 'Monthly', 'Rarely', 'First-Time'].map((freq) => (
            <option value={freq} key={freq}>{freq}</option>
          ))}
        </select>
        <input
          type="text"
          placeholder="Feedback Score"
          {...formik.getFieldProps('feedbackScore')}
          className="form-control mb-3"
        />
        <input
          type="text"
          placeholder="Loyalty Level"
          {...formik.getFieldProps('loyaltyLevel')}
          className="form-control mb-3"
        />
        <button type="submit" className="btn btn-primary w-100">Submit Feedback</button>
      </form>
    </div>
  );
}
export default CustomerFeedback;