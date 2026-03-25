import React from "react";
import { COLORS } from "../utils/constants";

function LetterDisplay({
  currentLetter,
  confidence,
  currentWord,
  dynamicWord,
}) {
  const confidencePercent = Math.round((confidence || 0) * 100);
  const confidenceColor =
    confidencePercent > 80
      ? COLORS.success
      : confidencePercent > 60
        ? COLORS.warning
        : COLORS.danger;

  return (
    <div className="letter-display" aria-label="Recognition Details">
      {dynamicWord && (
        <div className="dynamic-word-overlay" aria-live="assertive">
          <span className="dynamic-label">DYNAMIC WORD:</span>
          <span className="dynamic-value">{dynamicWord.toUpperCase()}</span>
        </div>
      )}

      <div className="current-letter-section" aria-live="polite">
        <span className="label" style={{ color: COLORS.textSecondary }}>
          Active Recognition
        </span>
        <span
          className="big-letter"
          aria-label={`Detected letter: ${currentLetter && currentLetter !== "nothing" ? currentLetter : "none"}`}
        >
          {currentLetter && currentLetter !== "nothing"
            ? currentLetter.toUpperCase()
            : "—"}
        </span>
        <div
          className="confidence-container"
          aria-label={`Confidence score: ${confidencePercent}%`}
        >
          <div className="confidence-bar-container" aria-hidden="true">
            <div
              className="confidence-bar"
              style={{
                width: `${confidencePercent}%`,
                backgroundColor: confidenceColor,
                boxShadow: `0 0 10px ${confidenceColor}`,
              }}
            />
          </div>
          <span
            className="confidence-text"
            style={{ color: confidenceColor }}
            aria-hidden="true"
          >
            CONFIDENCE: {confidencePercent}%
          </span>
        </div>
      </div>

      <div className="current-word-section" aria-live="polite">
        <span className="label" style={{ color: COLORS.textSecondary }}>
          Buffer
        </span>
        <span
          className="current-word"
          aria-label={`Current word buffer: ${currentWord || "empty"}`}
        >
          {currentWord || "EMPTY"}
        </span>
      </div>
    </div>
  );
}

export default LetterDisplay;
