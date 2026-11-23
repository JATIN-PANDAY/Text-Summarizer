# AI Dialogue & Text Summarizer

An end-to-end NLP web application that generates concise, sentence-cased summaries from multi-turn dialogues and long-form passages using a fine-tuned **T5-small** Transformer model served via **FastAPI** and a **Bootstrap** frontend.

## Features
- **Fine-Tuned T5-small Inference:** Trained on the SAMSum dialogue summarization dataset using PyTorch and Hugging Face `Transformers`.
- **RESTful FastAPI Backend:** Exposes a `/summarize/` POST endpoint with input preprocessing, beam-search decoding (`num_beams=4`), and regex sentence-case post-processing.
- **Responsive Web Interface:** Clean Bootstrap card UI with asynchronous `fetch()` calls and real-time loading states.

## Tech Stack
- **Backend:** Python 3.12, FastAPI, Uvicorn, Pydantic, Jinja2
- **Machine Learning / NLP:** PyTorch, Hugging Face Transformers (`T5ForConditionalGeneration`, `T5Tokenizer`)
- **Frontend:** HTML5, CSS3, Bootstrap 5, Vanilla JavaScript

## Project Structure
```text
text-sumrizer/
├── app.py                  # FastAPI server, model inference & sentence formatting
├── index.html              # Bootstrap UI & async fetch client
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes large model weights & CSV datasets
└── saved_memory_model/     # Fine-tuned T5 model directory (local)