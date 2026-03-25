import React from "react";

function TextBox({ sentence }) {
  return (
    <div className="text-box" role="region" aria-live="polite" aria-atomic="false">
      <span className="text-box-label" id="translation-label">Live Translation Pipeline</span>
      <div className="text-box-content" aria-labelledby="translation-label">
        {sentence ? (
          <span>{sentence}</span>
        ) : (
          <span className="text-placeholder" aria-hidden="true">
            Waiting for input stream...
          </span>
        )}
        <span className="text-cursor-blink" aria-hidden="true">_</span>
      </div>
    </div>
  );
}

export default TextBox;
