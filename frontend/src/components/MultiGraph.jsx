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
    let t = 0;

    const interval = setInterval(() => {
      t++;

      const newPoint = {
        time: t,
      };

      // generate fake data for each line
      lines.forEach((line) => {
        newPoint[line] = generateValue(line);
      });

      setData((prev) => {
        const updated = [...prev, newPoint];
        if (updated.length > 20) updated.shift();
        return updated;
      });
    }, 1000);

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

/* fake realistic values */
function generateValue(type) {
  const base = {
    V: 230,
    I: 1,
    P: 50,
    VA: 300,
  };

  if (type.includes("V")) return base.V + Math.random() * 10 - 5;
  if (type.includes("I")) return base.I + Math.random() * 2;
  if (type.includes("VA")) return base.VA + Math.random() * 50;
  if (type.includes("W")) return base.P + Math.random() * 20;

  return Math.random() * 100;
}

const colors = ["#007AFF", "#34C759", "#FF9500"];

const card = {
  backgroundColor: "white",
  padding: "20px",
  borderRadius: "16px",
  boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
};