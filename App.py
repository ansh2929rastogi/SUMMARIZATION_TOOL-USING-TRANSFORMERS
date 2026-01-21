from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

# -------------------------------------------------
# INITIALIZE FASTAPI APP  
# -------------------------------------------------
app = FastAPI(
    title="Text Summarization System",
    description="Text summarization using FastAPI and T5",
    version="1.0"
)

# -------------------------------------------------
# DEVICE CONFIG
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# LOAD MODEL & TOKENIZER
# -------------------------------------------------
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
model.to(device)
model.eval()

# -------------------------------------------------
# TEMPLATES
# -------------------------------------------------
templates = Jinja2Templates(directory="Templates")

# -------------------------------------------------
# REQUEST SCHEMA
# -------------------------------------------------
class DialogueInput(BaseModel):
    dialogue: str

# -------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'\r\n|\n|\r', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# -------------------------------------------------
# SUMMARIZATION FUNCTION
# -------------------------------------------------
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_text(dialogue)

    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=180,
            min_length=60,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            length_penalty=1.0,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------------------------
# API ENDPOINT
# -------------------------------------------------
@app.post("/summarize/")
async def summarize(data: DialogueInput):
    summary = summarize_dialogue(data.dialogue)
    return {"summary": summary}

# -------------------------------------------------
# HTML UI
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
