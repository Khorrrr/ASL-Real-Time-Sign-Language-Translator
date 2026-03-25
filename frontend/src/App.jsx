import React, { useState, useEffect, useCallback, useRef } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import WebcamFeed from "./components/WebcamFeed";
import LetterDisplay from "./components/LetterDisplay";
import TextBox from "./components/TextBox";
import StatusBar from "./components/StatusBar";
import "./App.css";

function App() {
  const { isConnected, lastMessage, sendMessage } = useWebSocket();
  const [mode, setMode] = useState("alphabet");

  const [state, setState] = useState({
    handDetected: false,
    currentLetter: null,
    currentLetterConfidence: 0,
    acceptedLetter: null,
    currentWord: "",
    suggestions: [],
    suggestionBoxes: [],
    sentence: "",
    hover: null,
    indexTip: null,
    virtualCursor: null,
    isGrabbed: false,
    annotatedFrame: null,
    dynamicWord: null,
  });

  const [flashAccepted, setFlashAccepted] = useState(null);
  const isResetting = useRef(false);

  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === "update") {
        if (isResetting.current) return;

        setState((prev) => ({
          ...prev,
          handDetected: lastMessage.hasOwnProperty("hand_detected")
            ? lastMessage.hand_detected
            : prev.handDetected,
          currentLetter: lastMessage.hasOwnProperty("current_letter")
            ? lastMessage.current_letter
            : prev.currentLetter,
          currentLetterConfidence: lastMessage.hasOwnProperty(
            "current_letter_confidence",
          )
            ? lastMessage.current_letter_confidence
            : prev.currentLetterConfidence,
          currentWord: lastMessage.hasOwnProperty("current_word")
            ? lastMessage.current_word
            : prev.currentWord,
          suggestions: lastMessage.hasOwnProperty("suggestions")
            ? lastMessage.suggestions
            : prev.suggestions,
          sentence: lastMessage.hasOwnProperty("sentence")
            ? lastMessage.sentence
            : prev.sentence,
          hover: lastMessage.hasOwnProperty("hover")
            ? lastMessage.hover
            : prev.hover,
          indexTip: lastMessage.hasOwnProperty("index_tip")
            ? lastMessage.index_tip
            : prev.indexTip,
          virtualCursor: lastMessage.hasOwnProperty("virtual_cursor")
            ? lastMessage.virtual_cursor
            : prev.virtualCursor,
          isGrabbed: lastMessage.hasOwnProperty("dot_grabbed")
            ? lastMessage.dot_grabbed
            : prev.isGrabbed,
          dynamicWord: lastMessage.hasOwnProperty("dynamic_word")
            ? lastMessage.dynamic_word
            : prev.dynamicWord,
          suggestionBoxes: lastMessage.suggestion_boxes ?? prev.suggestionBoxes,
          annotatedFrame: lastMessage.annotated_frame ?? prev.annotatedFrame,
        }));

        if (lastMessage.accepted_letter) {
          setFlashAccepted(lastMessage.accepted_letter);
          setTimeout(() => setFlashAccepted(null), 800);
        }

        if (lastMessage.hover && lastMessage.hover.type === "accepted") {
          setFlashAccepted(`Word: ${lastMessage.hover.word}`);
          setTimeout(() => setFlashAccepted(null), 1500);
        }
      } else if (lastMessage.type === "reset_confirmed") {
        isResetting.current = false;
        setState((prev) => ({
          ...prev,
          currentWord: "",
          suggestions: [],
          suggestionBoxes: [],
          sentence: "",
          hover: null,
          dynamicWord: null,
          currentLetter: null,
          currentLetterConfidence: 0,
          acceptedLetter: null,
        }));
      }
    }
  }, [lastMessage]);

  const handleReset = useCallback(() => {
    console.log("!!! RESET BUTTON CLICKED !!!");
    isResetting.current = true;

    setState((prev) => ({
      ...prev,
      sentence: "",
      currentWord: "",
      dynamicWord: null,
      suggestions: [],
      suggestionBoxes: [],
      hover: null,
      currentLetter: null,
      currentLetterConfidence: 0,
      acceptedLetter: null,
    }));

    sendMessage({
      type: "reset",
      timestamp: Date.now(),
    });

    setTimeout(() => {
      if (isResetting.current) {
        console.warn("Reset confirmation timeout - unlocking updates");
        isResetting.current = false;
      }
    }, 3000);
  }, [sendMessage]);

  const toggleMode = (newMode) => {
    setMode(newMode);
    sendMessage({ type: "set_mode", mode: newMode });
  };

  const hoveredWord =
    state.hover && state.hover.type === "suggestion" ? state.hover.word : null;

  const hoverProgress = state.hover ? state.hover.progress : 0;

  return (
    <div className="app-container">
      <main className="main-layout">
        <section
          className="webcam-section"
          aria-label="Webcam feed and gesture recognition"
        >
          <WebcamFeed
            sendMessage={sendMessage}
            annotatedFrame={state.annotatedFrame}
            isConnected={isConnected}
            indexTip={state.indexTip}
            virtualCursor={state.virtualCursor}
            isGrabbed={state.isGrabbed}
            suggestions={state.suggestions}
            suggestionBoxes={state.suggestionBoxes}
            hoveredWord={hoveredWord}
            hoverProgress={hoverProgress}
          />

          <header className="floating-header">
            <nav className="mode-tabs" aria-label="Recognition mode">
              <button
                className={`tab-btn ${mode === "alphabet" ? "active" : ""}`}
                onClick={() => toggleMode("alphabet")}
              >
                Alphabet & Predictive
              </button>
              <button
                className={`tab-btn ${mode === "dynamic" ? "active" : ""}`}
                onClick={() => toggleMode("dynamic")}
              >
                Continuous Word
              </button>
            </nav>
            <TextBox sentence={state.sentence} />
          </header>

          <aside className="hud-side-panel">
            <LetterDisplay
              currentLetter={state.currentLetter}
              confidence={state.currentLetterConfidence}
              currentWord={state.currentWord}
              dynamicWord={state.dynamicWord}
            />
          </aside>

          <div className="floating-bottom-panel">
            <div className="controls-row">
              <button
                className="pill-btn danger"
                onClick={handleReset}
                aria-label="Reset translation pipeline"
              >
                Reset Session
              </button>
              <StatusBar
                isConnected={isConnected}
                handDetected={state.handDetected}
                acceptedLetter={flashAccepted}
              />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
