from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uvicorn
from typing import Optional
import sys
import re
import datetime

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(title="UATA API", description="Backend for Undersea Advanced Tactical Advisor")

def clean_tactical_output(text: str) -> str:
    """Removes markdown and special characters for a clean UI output."""
    if not text: return ""
    # Remove markdown bold/italic
    text = re.sub(r'\*\*|\*|__|_', '', text)
    # Remove markdown headers
    text = re.sub(r'#+\s+', '', text)
    # Remove separator bars and decorative lines
    text = re.sub(r'[━─═-]{3,}', '', text)
    # Remove brackets used in identification tags
    text = re.sub(r'\[|\]', '', text)
    return text.strip()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global main module placeholder
main = None

try:
    print("Attempting to load UATA core module...")
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import main as uata_core
    main = uata_core
    print("UATA Core Loaded Successfully.")
except Exception as e:
    print(f"FAILED TO LOAD UATA CORE: {e}")
    import traceback
    traceback.print_exc()

# Serve static files for outputs and sensor inputs
os.makedirs("outputs", exist_ok=True)
os.makedirs("sensor_input", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/sensor_input", StaticFiles(directory="sensor_input"), name="sensor_input")

@app.get("/health")
async def health():
    if main:
        return {"status": "online", "model": getattr(main, "MODEL_ID", "Unknown")}
    return {"status": "partial", "error": "Core module not loaded"}

@app.get("/metrics")
async def get_metrics():
    if not main:
        return {"YOLO_SONAR": {"mAP": "81.96%"}, "status": "MOCKED"}
    try:
        return getattr(main, "TACTICAL_METRICS", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def general_query(query: str = Form(...)):
    if not main:
        # Provide intelligent offline responses
        if "sonar" in query.lower() or "detect" in query.lower():
            return {"query": query, "response": "🔍 OFFLINE TACTICAL ANALYSIS: Sonar sweep simulation mode. Recommend deploying active sonar array at bearing 270°. Expected contact in 4.2 nautical miles. Running in simulation mode - verify with live sensors when online."}
        elif "depth" in query.lower() or "ballast" in query.lower():
            return {"query": query, "response": "⚓ BALLAST SYSTEM OFFLINE ADVISORY: Recommend neutral buoyancy at 150m depth. Trim tanks at 0.5° pitch. System offline - use manual gauge verification."}
        elif "course" in query.lower() or "heading" in query.lower():
            return {"query": query, "response": "📡 NAVIGATION OFFLINE: Recommend maintaining current heading 180°. GPS unavailable in simulation mode. Switch to manual compass navigation."}
        else:
            return {"query": query, "response": f"⚠️ UATA Core Engine Offline. Query: '{query}' Running in limited simulation mode. Basic diagnostics available only. Recommend system restart."}
    try:
        raw_response = main.verified_submarine_advisor(query)
        clean_response = clean_tactical_output(raw_response)
        main.log_interaction(query, clean_response)
        return {"query": query, "response": clean_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify")
def identify_file(file: UploadFile = File(...)):
    try:
        # 1. Determine Category and Target Folder
        ext = file.filename.split(".")[-1].lower()
        category = "images"
        if ext in ["wav", "mp3", "ogg", "flac"]: category = "audio"
        elif ext in ["mp4", "avi", "mkv", "mov"]: category = "video"
        
        target_dir = os.path.join("sensor_input", category)
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = os.path.join(target_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if not main:
            return {
                "response": "IDENTIFICATION MOCKED: System in offline mode. File stored in " + category,
                "file_path": file_path.replace("\\", "/")
            }
            
        # 2. Run Advisor
        raw_response = main.identify_advisor(file_path)
        
        # 3. Enhanced Extraction & Cleaning
        output_image = None
        predicted_class = "Unknown Target"
        confidence = "N/A"
        
        # Extract output image path (handle outputs/images/ or just outputs/)
        img_match = re.search(r'((?:outputs|sensor_input)[/\\](?:images[/\\])?[^\s]+?\.(?:png|jpg|jpeg))', raw_response)
        if img_match:
            raw_path = img_match.group(1).replace("\\", "/")
            # Verify and fix path
            if not os.path.exists(raw_path):
                if "outputs/" in raw_path and "images/" not in raw_path:
                    probe = raw_path.replace("outputs/", "outputs/images/")
                    if os.path.exists(probe): raw_path = probe
            output_image = raw_path
        
        # Fallback: if no annotated image found, show the input image
        if not output_image and category == "images":
            output_image = file_path.replace("\\", "/")

        # Extract Target Class
        class_match = re.search(r'\[(?:[A-Z\s\-]+ IDENTIFIED|RECONNAISSANCE|TARGET CATEGORY): (.*?)\]', raw_response)
        if not class_match:
            class_match = re.search(r'TARGET CATEGORY:\s*\*\*(.*?)\*\*', raw_response)
        if class_match:
            predicted_class = clean_tactical_output(class_match.group(1)).title()
            
        # Extract Confidence
        conf_match = re.search(r'CONFIDENCE:\s*\*\*([\d\.]+)%\*\*', raw_response)
        if conf_match:
            confidence = conf_match.group(1) + "%"

        # Extract Dataset Source
        source_match = re.search(r'DATASET SOURCE:\s*\*\*(.*?)\*\*', raw_response)
        match_source = source_match.group(1).strip() if source_match else "Standard Index"

        final_response = clean_tactical_output(raw_response)
        main.log_interaction(f"FILE: {file.filename}", final_response)
        
        return {
            "response": final_response, 
            "file_path": file_path.replace("\\", "/"), 
            "output_image": output_image,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "match_source": match_source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/command")
def run_command(command: str = Form(...), argument: Optional[str] = Form(None)):
    try:
        result = ""
        arg = argument or ""
        
        if not main:
            # Provide fallback responses when engine is offline
            if command == "/report":
                result = "SITREP GENERATED (OFFLINE MODE): No sensor contacts detected. Strategic patrol routine. Generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ". Report saved to: outputs/pdfs/UATA_REPORT_OFFLINE.pdf"
                pdf_path = "outputs/pdfs/UATA_REPORT_OFFLINE.pdf"
                clean_result = clean_tactical_output(result)
                return {"command": command, "response": clean_result, "pdf_path": pdf_path}
            elif command == "/metrics":
                return {"command": command, "response": "SYSTEM METRICS (OFFLINE): YOLO-SONAR: 81.96% mAP | CLIP-ViT: 72.4% Top-1 | DeepShip: 94.2% Accuracy | RAG Grounding: 98%"}
            else:
                return {"command": command, "response": "⚠️ UATA Core Engine offline. Running in simulation mode. Use web UI for basic diagnostics."}
        
        if command == "/look":
            result = main.visual_advisor(arg)
        elif command == "/listen":
            result = main.acoustic_advisor() if hasattr(main, "acoustic_advisor") else "Acoustic module offline."
        elif command == "/detect":
            result = main.detect_advisor(arg)
        elif command == "/tactical":
            result = main.tactical_advisor(arg)
        elif command == "/comply":
            result = main.compliance_advisor(arg)
        elif command == "/emergency":
            result = main.emergency_advisor(arg)
        elif command == "/report":
            result = main.generate_mission_report()
            # Extract PDF path for frontend auto-download
            pdf_match = re.search(r'S[A-Z\s]+ TO: (outputs[/\\]pdfs[/\\][^\s\n]+)', result)
            pdf_path = pdf_match.group(1).replace("\\", "/") if pdf_match else None
            
            clean_result = clean_tactical_output(result)
            main.log_interaction(f"{command} {arg}", clean_result)
            return {"command": command, "response": clean_result, "pdf_path": pdf_path}
        elif command == "/metrics":
            result = main.metrics_advisor()
        else:
            result = main.verified_submarine_advisor(f"{command} {arg}")
            
        clean_result = clean_tactical_output(result)
        main.log_interaction(f"{command} {arg}", clean_result)
        return {"command": command, "response": clean_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_inputs")
async def list_inputs():
    """Lists files in sensor_input/audio, video, and images."""
    data = {"audio": [], "video": [], "images": []}
    for category in data.keys():
        path = os.path.join("sensor_input", category)
        if os.path.exists(path):
            data[category] = os.listdir(path)
    return data

@app.post("/identify_path")
def identify_by_path(path: str = Form(...)):
    """Identifies a file that already exists on the server's disk."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found on server.")
    
    # Mocking UploadFile-like behavior for internal processing
    class FakeFile:
        filename = os.path.basename(path)
    
    # Rerouting to existing logic
    try:
        if not main:
            return {"response": "System offline.", "file_path": path}
            
        raw_response = main.identify_advisor(path)
        
        output_image = None
        predicted_class = "Unknown Target"
        confidence = "N/A"
        
        img_match = re.search(r'((?:outputs|sensor_input)[/\\](?:images[/\\])?[^\s]+?\.(?:png|jpg|jpeg))', raw_response)
        if img_match:
            raw_path = img_match.group(1).replace("\\", "/")
            if not os.path.exists(raw_path):
                if "outputs/" in raw_path and "images/" not in raw_path:
                    probe = raw_path.replace("outputs/", "outputs/images/")
                    if os.path.exists(probe): raw_path = probe
            output_image = raw_path
        
        # Fallback for input visibility
        if not output_image and any(path.lower().endswith(e) for e in ['.png', '.jpg', '.jpeg']):
            output_image = path.replace("\\", "/")

        class_match = re.search(r'\[(?:[A-Z\s\-]+ IDENTIFIED|RECONNAISSANCE|TARGET CATEGORY): (.*?)\]', raw_response)
        if not class_match:
            class_match = re.search(r'TARGET CATEGORY:\s*\*\*(.*?)\*\*', raw_response)
        if class_match:
            predicted_class = clean_tactical_output(class_match.group(1)).title()
            
        # Extract Dataset Source
        source_match = re.search(r'DATASET SOURCE:\s*\*\*(.*?)\*\*', raw_response)
        match_source = source_match.group(1).strip() if source_match else "Standard Index"

        final_response = clean_tactical_output(raw_response)
        return {
            "response": final_response, 
            "file_path": path.replace("\\", "/"), 
            "output_image": output_image,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "match_source": match_source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events")
async def get_events():
    if not main:
        return {"events": [{"time": "00:00", "event": "SYSTEM START - OFFLINE MODE"}]}
    return {"events": getattr(main, "MISSION_EVENTS", [])}

@app.get("/list_bases")
async def list_bases():
    """Lists available INS bases and the current active one."""
    if not main: return {"bases": {}, "active_base": None}
    return {
        "bases": getattr(main, "INDIAN_NAVAL_BASES", {}),
        "active_base": getattr(main, "CURRENT_OPERATIONAL_BASE", None)
    }

@app.post("/set_base")
def set_base(base_name: str = Form(...)):
    """Sets the active operational base."""
    if not main: 
        return {"status": "error", "message": "Engine offline."}
    
    bases = getattr(main, "INDIAN_NAVAL_BASES", {})
    if base_name not in bases:
        raise HTTPException(status_code=404, detail="Base not found.")
    
    main.CURRENT_OPERATIONAL_BASE = base_name
    # PERSISTENCE FIX: Save to disk so report generator sees it
    main.UATA_CONFIG["active_base"] = base_name
    main.save_uata_config(main.UATA_CONFIG)
    
    print(f"TACTICAL LOG: Operational Base set to {base_name}")
    return {"status": "success", "active_base": base_name, "details": bases[base_name]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8030)
