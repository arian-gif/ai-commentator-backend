from fastapi import FastAPI, UploadFile, File
import uuid
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import os
from dotenv import load_dotenv
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from collections import Counter
import json
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")
debug_mode = os.getenv("DEBUG", "False") == "True"

genai.configure(api_key=API_KEY)
client = genai.GenerativeModel("gemini-2.0-flash")

model = whisper.load_model("tiny")  # or "small", "medium"

mobilenet = mobilenet_v2(pretrained=True)
mobilenet.eval()

with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx_to_label = {int(key): value[1] for key, value in class_idx.items()}

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def classify_frame(frame):
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = mobilenet(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    return idx_to_label[top_catid.item()]

def extract_keyframes(video_path, frame_step=10, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = []
    for i in range(0, frame_count, frame_step):
        if len(selected_frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        selected_frames.append(frame)
    cap.release()
    return selected_frames

def summarize_predictions(predictions, top_n=3):
    counter = Counter(predictions)
    most_common = counter.most_common(top_n)
    description = ", ".join([f"{label} ({count} times)" for label, count in most_common])
    return f"The video contains the following main objects or actions: {description}."

@app.get("/")
async def home():
    print("hello from Fast")

@app.post("/upload/")
async def generate_commentary(file: UploadFile = File(...)):
    # Step 1: Save uploaded video
    video_id = str(uuid.uuid4())
    video_path = f"temp/{video_id}.mp4"
    audio_path = f"temp/{video_id}.wav"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Step 2: Extract audio and check if present
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        transcript = None
    else:
        clip.audio.write_audiofile(audio_path)
        result = model.transcribe(audio_path)
        transcript = result["text"].strip()

    if not transcript:
        frames = extract_keyframes(video_path)
        preds = [classify_frame(frame) for frame in frames]
        summary_text = summarize_predictions(preds)
        prompt = (
            "You're a sports commentator. The user uploaded a highlight video with no audio. "
            "Please generate exciting generic sports commentary this one clip. "
            f"Visual analysis summary: {summary_text}"
        )
    else:
        prompt = (
            f"You're a dramatic sports commentator. Here's a transcription of the play: \n\n{transcript}\n\n"
            "Generate hype sports commentary based on this."
        )

    response = client.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 300
        }
    )
    commentary = response.text
    return {"commentary":commentary}

