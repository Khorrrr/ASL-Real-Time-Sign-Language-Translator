import React from "react";

function HandCursor({ indexTip, virtualCursor, isGrabbed }) {
  const activeCursor = virtualCursor || indexTip;
  const isVirtual = !!virtualCursor;

  if (!activeCursor) return null;

  const cursorColor = isVirtual ? "#dc2626" : "var(--primary)";

  return (
    <div
      className={`hand-cursor ${isVirtual ? "is-virtual" : "is-passive"} ${isGrabbed ? "is-grabbed" : ""}`}
      style={{
        left: `${activeCursor.x * 100}%`,
        top: `${activeCursor.y * 100}%`,
        color: cursorColor,
      }}
    >
      <div className="hand-cursor-dot" />
      <div className="hand-cursor-ring" />
    </div>
  );
}

export default HandCursor;
