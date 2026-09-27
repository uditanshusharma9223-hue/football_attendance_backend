from fastapi import FastAPI, File, UploadFile, Form
import os
import shutil
import csv
from datetime import datetime
from deepface import DeepFace

app = FastAPI()

DB_PATH = "players_db"
CSV_FILE = "attendance_records.csv"
os.makedirs(DB_PATH, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Player Name", "Status"])

@app.post("/register-player")
async def register_player(name: str = Form(...), file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1]
    player_image_path = os.path.join(DB_PATH, f"{name}.{file_extension}")
    
    with open(player_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    for item in os.listdir(DB_PATH):
        if item.endswith(".pkl"):
            os.remove(os.path.join(DB_PATH, item))

    return {"status": "success", "message": f"{name} registered successfully!"}

@app.post("/group-attendance")
async def group_attendance(file: UploadFile = File(...)):
    temp_group_path = "temp_group.jpg"
    with open(temp_group_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    all_registered = [
        os.path.splitext(f)[0] for f in os.listdir(DB_PATH) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    present_players = set()

    try:
        results = DeepFace.find(
            img_path=temp_group_path,
            db_path=DB_PATH,
            model_name="VGG-Face",
            detector_backend="retinaface",
            enforce_detection=False,
            distance_metric="cosine"
        )

        for df in results:
            if not df.empty:
                for match_path in df['identity']:
                    filename = os.path.basename(match_path)
                    player_name = os.path.splitext(filename)[0]
                    present_players.add(player_name)

    except Exception as e:
        return {"status": "error", "message": str(e)}

    present_list = list(present_players)
    absent_list = [p for p in all_registered if p not in present_players]

    now = datetime.now()
    curr_date = now.strftime("%Y-%m-%d")
    curr_time = now.strftime("%H:%M:%S")

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for player in present_list:
            writer.writerow([curr_date, curr_time, player, "Present"])
        for player in absent_list:
            writer.writerow([curr_date, curr_time, player, "Absent"])

    return {
        "status": "success",
        "date": curr_date,
        "time": curr_time,
        "present_players": present_list,
        "absent_players": absent_list,
        "total_registered": len(all_registered),
        "total_present": len(present_list)
    }