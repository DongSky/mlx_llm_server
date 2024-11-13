import mlx.core as mx
from mlx_lm import load, generate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


model_name = "Qwen/Qwen2.5-3B-Instruct"
model, tokenizer = load(model_name)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    context: Optional[List[str]] = None
    stream: Optional[bool] = False
    raw: Optional[bool] = False
    format: Optional[str] = None


class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: List[int]
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


def generate_response(request: GenerateRequest):
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    
    # 添加历史对话信息
    if request.context:
        for i, content in enumerate(request.context):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": content})
    
    # 添加当前用户的提问
    messages.append({"role": "user", "content": request.prompt})
    
    print("Messages:", messages, flush=True)
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = generate(model, tokenizer, prompt=prompt, verbose=True)

    return GenerateResponse(
        model=request.model,
        created_at="",  # You might want to add a timestamp here
        response=outputs,
        done=True,
        context=[],  # MLX doesn't provide context in the same way as Ollama
        total_duration=0,  # These durations are not provided by MLX
        load_duration=0,
        prompt_eval_count=0,
        prompt_eval_duration=0,
        eval_count=0,
        eval_duration=0
    )


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_api(request: GenerateRequest):
    # if request.model != model_name:
    #     raise HTTPException(status_code=400, detail="Requested model not available")
    return generate_response(request)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3301)
