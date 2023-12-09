from transformers import pipeline
from fastapi import FastAPI
import torch
from typing import List
import numpy as np
import random
from pydantic import BaseModel
import uvicorn
import argparse


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


class Data(BaseModel):
    prompt: str = "an image of"
    seed: int = 0
    max_length: int = 77
    additional_params: dict = {}


app = FastAPI()

generator = pipeline(model="Gustavosta/MagicPrompt-Stable-Diffusion", device="cuda")

@app.post("/prompt_generate")
async def get_rewards(data: Data):
    seed_everything(data.seed)
    prompt = generator(
        data.prompt,
        max_length=data.max_length,
        **data.additional_params,
    )[0]["generated_text"]
    print("Prompt Generated:", prompt, flush=True)
    return {"prompt": prompt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10001)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
