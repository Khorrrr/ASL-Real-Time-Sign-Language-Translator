import React from "react";

function SuggestionBar({ suggestionBoxes, hoveredWord, hoverProgress }) {
  if (!suggestionBoxes || suggestionBoxes.length === 0) {
    return null;
  }

  return (
    <>
      {suggestionBoxes.map((box, index) => {
        const isHovered = hoveredWord === box.word;

        const style = {
          position: "absolute",
          left: `${box.x_min * 100}%`,
          top: `${box.y_min * 100}%`,
          width: `${(box.x_max - box.x_min) * 100}%`,
          height: `${(box.y_max - box.y_min) * 100}%`,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          pointerEvents: "none",
        };

        return (
          <div
            key={index}
            className={`suggestion-box ${isHovered ? "hovered" : ""}`}
            style={style}
          >
            <span className="suggestion-word">{box.word.toUpperCase()}</span>
            {isHovered && (
              <div
                className="hover-progress-bar"
                style={{
                  width: "80%",
                  height: "4px",
                  background: "rgba(255,255,255,0.2)",
                  marginTop: "4px",
                  borderRadius: "2px",
                  overflow: "hidden",
                }}
              >
                <div
                  className="hover-progress-fill"
                  style={{
                    width: `${(hoverProgress || 0) * 100}%`,
                    height: "100%",
                    backgroundColor: "white",
                    transition: "width 0.1s linear",
                  }}
                />
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}

export default SuggestionBar;
