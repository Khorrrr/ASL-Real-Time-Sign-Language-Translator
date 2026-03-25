import React, { useRef, useEffect, useState } from "react";
import HandCursor from "./HandCursor";
import SuggestionBar from "./SuggestionBar";

function WebcamFeed({
  sendMessage,
  annotatedFrame,
  isConnected,
  indexTip,
  virtualCursor,
  isGrabbed,
  suggestions,
  suggestionBoxes,
  hoveredWord,
  hoverProgress,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [overlayStyle, setOverlayStyle] = useState({});

  useEffect(() => {
    const el = imageRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        setOverlayStyle({
          width: `${width}px`,
          height: `${height}px`,
          position: "absolute",
          top: `${el.offsetTop}px`,
          left: `${el.offsetLeft}px`,
          pointerEvents: "none",
        });
      }
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, [cameraReady, annotatedFrame]);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setCameraReady(true);
        }
      } catch (err) {
        console.error("Camera error:", err);
      }
    }
    startCamera();
  }, []);

  // Capture and send frames to backend
  useEffect(() => {
    if (!isConnected || !cameraReady) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    let running = true;

    const capture = () => {
      if (!running) return;

      const videoW = video.videoWidth || 640;
      const videoH = video.videoHeight || 480;

      canvas.width = 700;
      canvas.height = 700 * (videoH / videoW);

      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      const imageData = canvas.toDataURL("image/jpeg", 0.4);
      sendMessage({
        type: "frame",
        image: imageData,
        timestamp: Date.now(),
      });

      setTimeout(capture, 60);
    };

    setTimeout(capture, 500);
    return () => {
      running = false;
    };
  }, [isConnected, cameraReady, sendMessage]);

  return (
    <div className="webcam-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ position: "absolute", opacity: 0, pointerEvents: "none" }}
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {annotatedFrame ? (
        <img
          ref={imageRef}
          src={annotatedFrame}
          alt="Neural Feed"
          className="webcam-image"
        />
      ) : cameraReady ? (
        <video
          ref={(el) => {
            if (el && videoRef.current)
              el.srcObject = videoRef.current.srcObject;
            imageRef.current = el;
          }}
          autoPlay
          playsInline
          muted
          className="webcam-image"
          style={{ transform: "scaleX(-1)" }}
        />
      ) : (
        <div className="webcam-placeholder">INITIALIZING SENSORS...</div>
      )}

      <div className="cursor-overlay-container" style={overlayStyle}>
        <HandCursor
          indexTip={indexTip}
          virtualCursor={virtualCursor}
          isGrabbed={isGrabbed}
        />
        <SuggestionBar
          suggestionBoxes={suggestionBoxes}
          hoveredWord={hoveredWord}
          hoverProgress={hoverProgress}
        />
      </div>
    </div>
  );
}

export default WebcamFeed;
