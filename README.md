# Closure

This is a chatbot project built using FastAPI and MLX framework, defaulting to the Qwen2.5-3B-Instruct model.

## Project Structure

- `main.py`: Backend server code, containing API endpoints and model loading logic
- `test_main.py`: Unit tests for the backend API
- `index.html`: Frontend interface

## Features

- Uses the Qwen2.5-3B-Instruct model for dialogue generation
- Supports multi-turn conversations, maintaining context
- Provides a function to clear context
- Clean Material Design style interface

## Installing Dependencies

```bash
pip install fastapi uvicorn mlx-lm
```

## Running the Project

```bash
python main.py
```

## Running the Frontend

```bash
python -m http.server 8000
```

## Accessing the API

```bash
curl http://127.0.0.1:8000/chat
```
