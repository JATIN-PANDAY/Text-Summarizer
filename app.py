from fastapi import FastAPI,Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initialize our fastapi app 

app = FastAPI(title="Text Summarizer", description="A simple text summarization API using T5 model", version="1.0.0")


model = T5ForConditionalGeneration.from_pretrained("./saved_memory_model/saved_memory_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_memory_model/saved_memory_model")

#device


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

# templating

templates = Jinja2Templates(directory=".")

# input schema For dialogue



class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()



def format_sentences(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    # Capitalize the very first letter and any letter after ., !, or ? followed by spaces
    return re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        text,
    )


def summarize_dialogue(dialogue: str) -> str:
    cleaned = clean_data(dialogue)
    inputs = tokenizer(
        "summarize: " + cleaned,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    raw_summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return format_sentences(raw_summary)
# API Endpoints


@app.post("/summarize/")

async def create_item(dialogue_input:DialogueInput):
    summary=summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)

async def home(request:Request):
    return templates.TemplateResponse(request=request, name="index.html")
