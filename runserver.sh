conda activate eds
cd ~\Documents\workspaces\emotion_detection
uvicorn server:app --reload --host=192.168.0.41 --port=8000