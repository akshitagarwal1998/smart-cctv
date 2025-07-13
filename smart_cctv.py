import cv2
import time
import sqlite3
import argparse
import torch
from transformers import pipeline
from datetime import datetime
from PIL import Image 

# --- LLM/CLIP imports
import openai
from transformers import CLIPModel, CLIPProcessor

import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# List of zero-shot categories or activities to detect; modify as needed
CANDIDATE_LABELS = ["person", "car"]
DETECTION_FRAMES = 5  # frames between runs

BOX_THRESHOLD = 0.1    # minimum box confidence
TEXT_THRESHOLD = 0.1   # minimum text match score

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

# System prompt for the journal-generating LLM; customize to your style
SYSTEM_PROMPT = "You are a security log assistant that writes concise, timestamped journal entries of observed events."
JOURNAL_MODEL = os.getenv("JOURNAL_MODEL", "gpt-3.5-turbo")  # LLM model for journal; override via env

DB_PATH = "detections.db"
TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    label TEXT NOT NULL,
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
    frame_index INTEGER NOT NULL
);
"""

# -----------------------------------------------------------------------------
# DATABASE HELPERS
# -----------------------------------------------------------------------------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(TABLE_SCHEMA)
    conn.commit()
    return conn

def log_detection(conn, timestamp, label, box, frame_idx):
    # box may be a dict with keys xmin/ymin/xmax/ymax or a sequence [x0,y0,x1,y1]
    if isinstance(box, dict):
        x0 = int(box.get("xmin", box.get("x0", 0)))
        y0 = int(box.get("ymin", box.get("y0", 0)))
        x1 = int(box.get("xmax", box.get("x1", 0)))
        y1 = int(box.get("ymax", box.get("y1", 0)))
    else:
        x0, y0, x1, y1 = map(int, box)
    conn.execute(
        "INSERT INTO detections (timestamp, label, x0,y0,x1,y1, frame_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (timestamp, label, x0, y0, x1, y1, frame_idx)
    )
    conn.commit()
    # Log the insertion to console
    logger.info(f"Inserted detection into DB: timestamp={timestamp}, label={label}, box=({x0},{y0},{x1},{y1}), frame_index={frame_idx}")

# -----------------------------------------------------------------------------
# LLM/CRITICALITY AGENTS
# -----------------------------------------------------------------------------
class LLMJournalAgent:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_journal(self, events):
        # Build a detailed prompt for narrative reporting
        prompt = (
            "Based on the following detection events, write a detailed security journal entry. "
            "Include potential safety concerns (e.g., collisions, thefts), describe what might be happening, "
            "and use vivid, descriptive language:\n"
        )
        for e in events:
            prompt += f"- {e['timestamp']}: {e['label']} detected at frame {e['frame_idx']} (event: {e.get('event', 'N/A')})\n"
        # Call the LLM
        resp = openai.chat.completions.create(
            model=JOURNAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content

class CriticalityAgent:
    def __init__(self):
        self.label_sev = {"person":2,"car":3,"bag":2,"bottle":1,"phone":2,"laptop":3}
        self.event_wt = {"appeared":1,"reappeared":1,"disappeared":4}

    def analyze(self, events):
        critical = []
        for e in events:
            lvl = self.label_sev.get(e['label'],1) + self.event_wt.get("appeared",1)
            if lvl >= 4:
                critical.append((e, lvl))
        return critical

# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
def main(source, device, interval):
    # 1) Initialize DB
    conn = init_db()

    # 2) Initialize HF zero-shot detector
    detect_pipe = pipeline(
        "zero-shot-object-detection",
        model="IDEA-Research/grounding-dino-tiny",
        device=0 if device.startswith("cuda") else -1,
        torch_dtype=torch.float32,
    )

    # 3) OpenCV capture
    cap = cv2.VideoCapture(0 if source is None else source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open capture source {source}")

    frame_idx = 0

    # --- LLM/criticality agents and buffer
    journal_agent = LLMJournalAgent(OPENAI_API_KEY)
    critical_agent = CriticalityAgent()
    buffer_events = []

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        logger.debug(f"Processing frame {frame_idx}")

        if frame_idx % interval == 0:
            # convert BGR frame → PIL RGB image
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # run detection on PIL image
            results = detect_pipe(
                pil,
                candidate_labels=CANDIDATE_LABELS,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            logger.info(f"Frame {frame_idx}: Detected {len(results)} objects")
            ts = datetime.now().isoformat(sep=" ", timespec="seconds")
            for r in results:
                label = r["label"]
                score = r["score"]
                box   = r["box"]  # [x0,y0,x1,y1]
                # Optionally filter by score threshold:
                # if score < 0.3: continue
                log_detection(conn, ts, label, box, frame_idx)
                buffer_events.append({"timestamp":ts, "label":label, "frame_idx":frame_idx, "event":"appeared"})
                # Draw on frame
                # Unpack box whether it’s a dict or list
                if isinstance(box, dict):
                    x0 = int(box.get("xmin", box.get("x0", 0)))
                    y0 = int(box.get("ymin", box.get("y0", 0)))
                    x1 = int(box.get("xmax", box.get("x1", 0)))
                    y1 = int(box.get("ymax", box.get("y1", 0)))
                else:
                    x0, y0, x1, y1 = map(int, box)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}:{score:.2f}", (x0, y0 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Generate journal
            journal_text = journal_agent.generate_journal(buffer_events)
            logger.info(f"Journal entry:\n{journal_text}")

            # Analyze criticality
            critical_list = critical_agent.analyze(buffer_events)
            if critical_list:
                for e,lvl in critical_list:
                    logger.warning(f"Critical event level {lvl}: {e['timestamp']} {e['label']} at frame {e['frame_idx']}")

            # Clear buffer for next interval
            buffer_events.clear()

        # Throttle to target FPS
        elapsed = time.time() - loop_start
        target_fps = 1.0
        delay = max(0.0, (1.0/target_fps) - elapsed)
        time.sleep(delay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    conn.close()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# ARGPARSE & ENTRYPOINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart CCTV with Grounding DINO zero-shot detection + SQLite logging"
    )
    parser.add_argument(
        "--video", "-v", type=str, default=None,
        help="Path to video file; if omitted, uses webcam (0)"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=DETECTION_FRAMES,
        help="Number of frames between detections"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cpu",
        help="Torch device: 'cpu' or 'cuda'"
    )
    args = parser.parse_args()

    main(args.video, args.device, args.interval)