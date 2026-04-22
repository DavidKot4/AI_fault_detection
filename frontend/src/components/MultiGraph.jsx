import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

export default function MultiGraph({ title, lines }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:5000/data")
        .then((res) => res.json())
        .then((newData) => {
          if (newData.error) {
            console.error("Backend error:", newData.error);
            return;
          }

          const point = {
            time: newData.timestamp,
          };

          // map backend values into graph
          lines.forEach((line) => {
            point[line] = newData[line];
          });

          setData((prev) => {
            const updated = [...prev, point];
            if (updated.length > 20) updated.shift(); // keep last 20 points
            return updated;
          });
        })
        .catch((err) => console.error(err));
    }, 1000); // update every second

    return () => clearInterval(interval);
  }, [lines]);

  return (
    <div style={card}>
      <h3>{title}</h3>

      <div style={{ width: "100%", height: 250 }}>
        <ResponsiveContainer>
          <LineChart data={data}>
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />

            {lines.map((line, i) => (
              <Line
                key={line}
                type="monotone"
                dataKey={line}
                stroke={colors[i % colors.length]}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const colors = ["#007AFF", "#34C759", "#FF9500"];

const card = {
  backgroundColor: "white",
  padding: "20px",
  borderRadius: "16px",
  boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
};