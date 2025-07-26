import React, { useState, useEffect } from 'react';
import { Select, Card, CardContent, Button } from '@shadcn/ui';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const mockPredictionData = [
  { name: 'Jokić', Predicted: 28, Actual: 30 },
  { name: 'Embiid', Predicted: 31, Actual: 29 },
  { name: 'Giannis', Predicted: 26, Actual: 27 },
  { name: 'Doncic', Predicted: 29, Actual: 25 },
  { name: 'Tatum', Predicted: 24, Actual: 23 },
];

const mockFeatureImportance = [
  { feature: 'Win Shares', value: 0.35 },
  { feature: 'PTS/Game', value: 0.30 },
  { feature: 'AST/Game', value: 0.20 },
  { feature: 'REB/Game', value: 0.10 },
  { feature: 'Opp Def Rating', value: 0.05 },
];

export default function SportsInsightDashboard() {
  const [player, setPlayer] = useState('Jokic');
  const [season, setSeason] = useState('2024-25');
  const [data, setData] = useState(mockPredictionData);

  const handlePredict = () => {
    // placeholder for API call
    setData(mockPredictionData);
  };

  return (
    <div className="p-4 space-y-6">
      <h1 className="text-2xl font-semibold">SportsInsight MVP – Next-Game Prediction</h1>

      <div className="flex space-x-4">
        <Select value={player} onChange={setPlayer} className="w-1/4">
          <option value="Jokic">Nikola Jokic</option>
          <option value="Embiid">Joel Embiid</option>
          <option value="Giannis">Giannis Antetokounmpo</option>
          <option value="Doncic">Luka Doncic</option>
          <option value="Tatum">Jayson Tatum</option>
        </Select>
        <Select value={season} onChange={setSeason} className="w-1/4">
          <option value="2023-24">2023-24</option>
          <option value="2024-25">2024-25</option>
        </Select>
        <Button onClick={handlePredict} className="w-1/6">
          Predict
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card className="h-96 p-2">
          <CardContent>
            <h2 className="text-lg font-medium mb-2">Predicted vs Actual Points</h2>
            <ResponsiveContainer width="100%" height="80%">
              <BarChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Predicted" />
                <Bar dataKey="Actual" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="h-96 p-2">
          <CardContent>
            <h2 className="text-lg font-medium mb-2">Feature Importance</h2>
            <ResponsiveContainer width="100%" height="80%">
              <BarChart
                layout="vertical"
                data={mockFeatureImportance}
                margin={{ top: 20, right: 20, bottom: 20, left: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" />
                <Tooltip />
                <Bar dataKey="value" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}