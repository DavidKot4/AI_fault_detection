import React, { useState, useEffect } from "react";
import MultiGraph from "./MultiGraph";

export default function Dashboard() {
  const [fault, setFault] = useState("No Fault");
  const [confidence, setConfidence] = useState(0);
  const [time, setTime] = useState("");

  useEffect(() => {
    setTimeout(() => {
      setFault("1-Phase Fault");
      setConfidence(0.92);
      setTime("Apr 9, 14:30");
    }, 2000);
  }, []);

  const isNormal = fault === "No Fault";

  return (
    <div style={container}>
      {/* TITLE */}
      <h1 style={titleStyle}>⚡ Smart Grid Fault Monitor ⚡</h1>

      {/* STATUS (stays left like before) */}
      <div style={statusContainer}>
        <div
          style={{
            ...statusDot,
            backgroundColor: isNormal ? "#34C759" : "#FF3B30",
          }}
        />
        <span style={statusText}>
          {isNormal ? "System Normal" : "Fault Detected"}
        </span>
      </div>

      {/* CARDS */}
      <div style={cardRow}>
        <Card title="Fault Type" value={fault} />
        <Card
          title="Confidence"
          value={`${(confidence * 100).toFixed(2)}%`}
        />
        <Card title="Last Update" value={time} />
      </div>

      <div style={graphGrid}>
        <MultiGraph
          title="Voltages (Line-to-Neutral)"
          lines={["V_L1", "V_L2", "V_L3"]}
        />

        <MultiGraph
          title="Voltages (Line-to-Line)"
          lines={["V_L1_L2", "V_L2_L3", "V_L3_L1"]}
        />

        <MultiGraph
          title="Current"
          lines={["I_L1", "I_L2", "I_L3"]}
        />

        <MultiGraph
          title="Apparent Power"
          lines={["VA_L1", "VA_L2", "VA_L3"]}
        />

        <MultiGraph
          title="Active Power"
          lines={["W_L1", "W_L2", "W_L3"]}
        />
      </div>

      {/* HISTORY */}
      <div style={historyCard}>
        <h3 style={{ marginBottom: "10px" }}>Recent Activity</h3>
        <p style={historyItem}>1-Phase Fault • 92% • 14:30</p>
        <p style={historyItem}>2-Phase Fault • 87% • 14:10</p>
        <p style={historyItem}>No Fault • 100% • 13:55</p>
      </div>
    </div>
  );
}

/* CARD COMPONENT */
function Card({ title, value }) {
  return (
    <div style={card}>
      <p style={cardTitle}>{title}</p>
      <p style={cardValue}>{value}</p>
    </div>
  );
}

/* STYLES */

const container = {
  minHeight: "100vh",
  backgroundColor: "#F5F5F7",
  padding: "40px",
  fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
  color: "#1d1d1f",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
};

const titleStyle = {
  fontSize: "36px",
  fontWeight: "600",
  marginBottom: "20px",
  color: "#000000",
};

const statusContainer = {
  display: "flex",
  alignItems: "center",
  gap: "10px",
  width: "100%",
  maxWidth: "900px",
  marginBottom: "20px",
};

const statusDot = {
  width: "12px",
  height: "12px",
  borderRadius: "50%",
};

const statusText = {
  fontSize: "18px",
};

const cardRow = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  gap: "20px",
  flexWrap: "wrap",
  width: "100%",
  maxWidth: "900px",
};

const card = {
  backgroundColor: "white",
  padding: "20px",
  borderRadius: "16px",
  width: "220px",
  textAlign: "center",
  boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
};

const cardTitle = {
  fontSize: "14px",
  color: "#6e6e73",
  marginBottom: "8px",
};

const cardValue = {
  fontSize: "20px",
  fontWeight: "600",
};

const historyCard = {
  marginTop: "40px",
  backgroundColor: "white",
  padding: "20px",
  borderRadius: "16px",
  boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
  width: "100%",
  maxWidth: "900px",
};

const historyItem = {
  fontSize: "14px",
  color: "#3a3a3c",
  marginBottom: "6px",
};

const graphGrid = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "20px",
  marginTop: "40px",
  width: "100%",
  maxWidth: "1100px",
};