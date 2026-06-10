import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const API_URL = 'https://ecommerce-ai-platform-65vm.onrender.com';

function Dashboard() {
  const [forecast, setForecast] = useState(null);
  const [supportResponse, setSupportResponse] = useState('');
  const [message, setMessage] = useState('');

  useEffect(() => {
    axios.post(`${API_URL}/forecast/daily`, { days: 30 })
      .then(res => setForecast(res.data))
      .catch(err => console.error(err));
  }, []);

  const chartData = forecast?.dates.map((date, i) => ({
    date: date.slice(5),
    orders: forecast.predictions[i]
  }));

  const handleSupport = () => {
    axios.post(`${API_URL}/support`, { message })
      .then(res => setSupportResponse(res.data.response))
      .catch(err => console.error(err));
  };

  return (
    <div style={{ padding: 20, fontFamily: 'Arial' }}>
      <h1>Olist MLOps Platform</h1>
      
      <div style={{ display: 'flex', gap: 20, marginBottom: 30 }}>
        <div style={{ background: '#1a1a2e', color: 'white', padding: 20, borderRadius: 10 }}>
          <h3>30-Day Forecast</h3>
          <p style={{ fontSize: 24 }}>{forecast?.total_forecast?.toFixed(0)} orders</p>
        </div>
        <div style={{ background: '#e94560', color: 'white', padding: 20, borderRadius: 10 }}>
          <h3>Avg Daily</h3>
          <p style={{ fontSize: 24 }}>{forecast?.avg_daily?.toFixed(0)} orders</p>
        </div>
        <div style={{ background: '#0f3460', color: 'white', padding: 20, borderRadius: 10 }}>
          <h3>Model R²</h3>
          <p style={{ fontSize: 24 }}>0.70</p>
        </div>
      </div>

      <div style={{ background: '#f5f5f5', padding: 20, borderRadius: 10, marginBottom: 30 }}>
        <h3>30-Day Demand Forecast</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="orders" stroke="#1a1a2e" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ background: '#f5f5f5', padding: 20, borderRadius: 10 }}>
        <h3>AI Customer Support</h3>
        <input 
          value={message} 
          onChange={e => setMessage(e.target.value)}
          placeholder="Ask a question..."
          style={{ width: '70%', padding: 10, marginRight: 10 }}
        />
        <button onClick={handleSupport} style={{ padding: 10, background: '#1a1a2e', color: 'white' }}>
          Send
        </button>
        {supportResponse && (
          <div style={{ marginTop: 20, padding: 10, background: 'white', borderRadius: 5 }}>
            <strong>Response:</strong> {supportResponse}
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;