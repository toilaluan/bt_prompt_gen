from transformers import pipeline
from fastapi import FastAPI, Request, Response, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import torch
from typing import List
import numpy as np
import random
from pydantic import BaseModel
import uvicorn
import argparse
import requests
import time
import threading


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
@app.middleware("http")
async def filter_allowed_ips(request: Request, call_next):
    print(str(request.url))
    if (request.client.host not in ALLOWED_IPS) and (request.client.host != "127.0.0.1"):
        print(f"A unallowed ip:", request.client.host)
        return Response(content="You do not have permission to access this resource", status_code=403)
    response = await call_next(request)
    return response

@app.post("/prompt_generate", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
async def get_rewards(data: Data):
    seed_everything(data.seed)
    prompt = generator(
        data.prompt,
        max_length=data.max_length,
        **data.additional_params,
    )[0]["generated_text"]
    print("Prompt Generated:", prompt, flush=True)
    return {"prompt": prompt}

def define_allowed_ips(url):
    global ALLOWED_IPS
    ALLOWED_IPS = []
    while True:
        response = requests.get(f"{url}/get_allowed_ips")
        response = response.json()
        ALLOWED_IPS = response['allowed_ips']
        print("Updated allowed ips:", ALLOWED_IPS, flush=True)
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--subnet_outbound_url", type=str, default="http://20.210.111.232:10005")
    args = parser.parse_args()
    allowed_ips_thread = threading.Thread(target=define_allowed_ips, args=(args.subnet_outbound_url,))
    allowed_ips_thread.setDaemon(True)
    allowed_ips_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
