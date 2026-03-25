"""
FastAPI WebSocket server — connects all the pieces together.
"""

import asyncio
import base64
import json
import cv2
import numpy as np
import os
import sys
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ASL-Backend")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.hand_tracker import HandTracker
from backend.letter_recognizer import LetterRecognizer
from backend.dynamic_recognizer import DynamicRecognizer
from backend.stabilizer import LetterStabilizer
from backend.word_predictor import WordPredictor
from backend.hover_detector import HoverDetector

app = FastAPI(title="ASL Translator Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Loading models.")
hand_tracker = HandTracker(max_hands=2)
letter_recognizer = LetterRecognizer()
dynamic_recognizer = DynamicRecognizer()
word_predictor = WordPredictor(model_name="gpt2")
executor = ThreadPoolExecutor(max_workers=1)
logger.info("All models loaded!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")

    stabilizer = LetterStabilizer(required_frames=18, cooldown_frames=20)
    hover_detector = HoverDetector()

    state = {
        "mode": "alphabet",
        "dynamic_sequence": [],
        "dynamic_word": None,
        "dynamic_cooldown": 0,
        "current_letters": "",
        "sentence": "",
        "suggestions": [],
        "hands_lost_frames": 0,
        "start_time": asyncio.get_event_loop().time(),
        "last_reset_time": 0,
        "dot_pos": {"x": 0.5, "y": 0.5},
        "dot_grabbed": False,
        "grabbed_handedness": None,
        "multi_hand_memory": 0,
    }
    
    HANDS_LOST_THRESHOLD = 20
    MULTI_HAND_GRACE = 10
    SMOOTHING_ALPHA = 0.8
    suggestion_task = None

    async def get_suggestions_async(sentence, letters):
        """Run word prediction in background thread."""
        if not letters: return
        loop = asyncio.get_event_loop()
        try:
            new_suggestions = await loop.run_in_executor(executor, word_predictor.get_suggestions, sentence, letters, 5)
            state["suggestions"] = new_suggestions
            hover_detector.update_suggestion_boxes(new_suggestions)
            await websocket.send_text(json.dumps({
                "type": "update",
                "suggestions": new_suggestions,
                "current_word": state["current_letters"],
                "sentence": state["sentence"]
            }))
        except asyncio.CancelledError: pass
        except Exception as e: print(f"Async prediction error: {e}")

    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except WebSocketDisconnect: break
            except Exception: continue

            if message["type"] == "set_mode":
                state["mode"] = message.get("mode", "alphabet")
                state["dynamic_sequence"] = []
                state["dynamic_word"] = None
                continue

            if message["type"] == "frame":
                if "timestamp" in message and message["timestamp"] < state["last_reset_time"]:
                    continue

                try:
                    img_data = base64.b64decode(message["image"].split(",")[1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None: continue
                except Exception: continue

                ts = int((asyncio.get_event_loop().time() - state["start_time"]) * 1000)

                try:
                    landmarks, hands_data, annotated_frame, is_multi_hand = hand_tracker.process_frame(frame, ts)
                except Exception as e:
                    print(f"Tracking error: {e}")
                    continue

                virtual_cursor = None
                
                if state["mode"] == "alphabet" and len(hands_data) == 2:
                    state["multi_hand_memory"] = 0
                    virtual_cursor = state["dot_pos"]

                    if state["dot_grabbed"]:
                        grabbing_hand = next((h for h in hands_data if h["label"] == state["grabbed_handedness"]), None)
                        
                        if grabbing_hand and grabbing_hand["is_pinched"]:
                            px, py = grabbing_hand["pinch_pos"]
                            state["dot_pos"]["x"] = px
                            state["dot_pos"]["y"] = py
                        else:
                            state["dot_grabbed"] = False
                            state["grabbed_handedness"] = None
                    else:
                        for hand in hands_data:
                            if hand["is_pinched"]:
                                px, py = hand["pinch_pos"]
                                dist = np.sqrt((px - state["dot_pos"]["x"])**2 + (py - state["dot_pos"]["y"])**2)
                                if dist < 0.12: 
                                    state["dot_grabbed"] = True
                                    state["grabbed_handedness"] = hand["label"]
                                    break
                else:
                    state["dot_grabbed"] = False
                    state["grabbed_handedness"] = None
                    state["multi_hand_memory"] += 1
                    if state["multi_hand_memory"] > 15:
                        state["dot_pos"] = {"x": 0.5, "y": 0.5}

                response = {
                    "type": "update",
                    "hand_detected": landmarks is not None,
                    "current_letter": None,
                    "current_letter_confidence": 0,
                    "accepted_letter": None,
                    "current_word": state["current_letters"],
                    "suggestions": state["suggestions"],
                    "suggestion_boxes": hover_detector.suggestion_boxes,
                    "sentence": state["sentence"],
                    "hover": None,
                    "virtual_cursor": virtual_cursor,
                    "dynamic_word": state["dynamic_word"],
                    "index_tip": {"x": hands_data[0]["index_tip"][0], "y": hands_data[0]["index_tip"][1]} if hands_data else None
                }

                if landmarks:
                    state["hands_lost_frames"] = 0

                    if state["mode"] == "dynamic":
                        full_landmarks = np.zeros(126)
                        for i, hand in enumerate(hands_data):
                            if i >= 2: break
                            start_idx = i * 63
                            coords = []
                            for lm in hand["landmarks"]:
                                coords.extend([lm["x"], lm["y"], lm["z"]])
                            full_landmarks[start_idx : start_idx + 63] = coords
                        
                        state["dynamic_sequence"].append(full_landmarks)
                        
                        if len(state["dynamic_sequence"]) > 30:
                            state["dynamic_sequence"].pop(0)
                        
                        if state["dynamic_cooldown"] > 0:
                            state["dynamic_cooldown"] -= 1

                        if len(state["dynamic_sequence"]) == 30 and state["dynamic_cooldown"] == 0:
                            word, confidence = dynamic_recognizer.predict(state["dynamic_sequence"])
                            if confidence > 0.85:
                                state["dynamic_word"] = word
                                response["dynamic_word"] = word
                                state["sentence"] += word + " "
                                response["sentence"] = state["sentence"]
                                state["dynamic_sequence"] = []
                                state["dynamic_cooldown"] = 30
                    else:
                        if not is_multi_hand:
                            letter, confidence = letter_recognizer.predict(landmarks)
                            response["current_letter"] = letter
                            response["current_letter_confidence"] = round(confidence, 3)
                            
                            accepted = stabilizer.update(letter, confidence)
                            if accepted:
                                response["accepted_letter"] = accepted
                                if accepted == "del":
                                    if state["current_letters"]: state["current_letters"] = state["current_letters"][:-1]
                                    elif state["sentence"]:
                                        words = state["sentence"].strip().split()
                                        if words:
                                            words.pop()
                                            state["sentence"] = " ".join(words) + (" " if words else "")
                                elif accepted == "space":
                                    if state["current_letters"]:
                                        state["sentence"] += state["current_letters"] + " "
                                        state["current_letters"] = ""
                                        state["suggestions"] = []
                                else:
                                    state["current_letters"] += accepted.upper()

                                if state["current_letters"]:
                                    if suggestion_task: suggestion_task.cancel()
                                    suggestion_task = asyncio.create_task(get_suggestions_async(state["sentence"].strip(), state["current_letters"]))
                                else:
                                    state["suggestions"] = []
                                    hover_detector.update_suggestion_boxes([])

                                response["current_word"] = state["current_letters"]
                                response["sentence"] = state["sentence"]

                    cursor_for_hover = None
                    if virtual_cursor:
                        cursor_for_hover = [(virtual_cursor["x"], virtual_cursor["y"])]
                    elif hands_data:
                        cursor_for_hover = [hands_data[0]["index_tip"]]

                    if cursor_for_hover and state["suggestions"]:
                        hr = hover_detector.check_hover(cursor_for_hover, virtual_cursor is not None)
                        if hr:
                            response["hover"] = hr
                            if hr["type"] == "accepted":
                                state["sentence"] += hr["word"] + " "
                                state["current_letters"] = ""
                                state["suggestions"] = []
                                hover_detector.update_suggestion_boxes([])
                                stabilizer.reset()
                                response.update({
                                    "sentence": state["sentence"],
                                    "current_word": "",
                                    "suggestions": [],
                                    "hover": {"type": "accepted", "word": hr["word"]}
                                })
                
                else:
                    state["hands_lost_frames"] += 1
                    if state["hands_lost_frames"] >= HANDS_LOST_THRESHOLD:
                        if state["current_letters"]:
                            state["sentence"] += state["current_letters"] + " "
                            state["current_letters"] = ""
                            state["suggestions"] = []
                            hover_detector.update_suggestion_boxes([])
                            stabilizer.reset()
                            response.update({"sentence": state["sentence"], "current_word": "", "suggestions": []})
                        state["hands_lost_frames"] = 0

                try:
                    _, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    response["annotated_frame"] = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                    await websocket.send_text(json.dumps(response))
                except Exception: pass

            elif message["type"] == "reset":
                logger.info("RECEIVED RESET COMMAND FROM CLIENT")
                state["last_reset_time"] = message.get("timestamp", 0)
                if suggestion_task: suggestion_task.cancel()
                
                state.update({
                    "current_letters": "",
                    "sentence": "",
                    "suggestions": [],
                    "dynamic_sequence": [],
                    "dynamic_word": None,
                    "dynamic_cooldown": 0,
                    "hands_lost_frames": 0,
                    "dot_pos": {"x": 0.5, "y": 0.5},
                    "dot_grabbed": False,
                    "grabbed_handedness": None,
                    "multi_hand_memory": 0,
                })
                stabilizer.reset()
                hover_detector.update_suggestion_boxes([])
                await websocket.send_text(json.dumps({
                    "type": "reset_confirmed", 
                    "sentence": "", 
                    "current_word": "", 
                    "suggestions": [],
                    "dynamic_word": None,
                    "suggestion_boxes": [],
                    "hover": None,
                    "timestamp": message.get("timestamp", 0)
                }))

    except Exception as e:
        if not isinstance(e, WebSocketDisconnect):
            print(f"Socket error: {e}")
    finally:
        if suggestion_task: suggestion_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
