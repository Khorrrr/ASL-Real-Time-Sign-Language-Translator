import React from "react";
import { COLORS } from "../utils/constants";

function StatusBar({ isConnected, handDetected, acceptedLetter }) {
  return (
    <div className="status-bar" role="status" aria-live="polite">
      <div className="status-item" aria-label={`System status: ${isConnected ? "Online" : "Offline"}`}>
        <span
          className="status-dot"
          aria-hidden="true"
          style={{
            backgroundColor: isConnected ? COLORS.success : COLORS.danger,
            color: isConnected ? COLORS.success : COLORS.danger,
          }}
        />
        <span>System: {isConnected ? "ONLINE" : "OFFLINE"}</span>
      </div>

      <div className="status-item" aria-label={`Sensor status: ${handDetected ? "Active" : "Scanning"}`}>
        <span
          className="status-dot"
          aria-hidden="true"
          style={{
            backgroundColor: handDetected ? COLORS.success : COLORS.danger,
            color: handDetected ? COLORS.success : COLORS.danger,
          }}
        />
        <span>Sensor: {handDetected ? "ACTIVE" : "SCANNING..."}</span>
      </div>

      {acceptedLetter && (
        <div className="status-item accepted-flash" aria-live="assertive">
          <span>{acceptedLetter.toUpperCase()}</span>
        </div>
      )}

      <div className="status-instructions" aria-hidden="true">
        [ TWO HAND HOVER TO COMMIT WORD ]
      </div>
    </div>
  );
}

export default StatusBar;
