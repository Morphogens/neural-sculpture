from clip_sdf import clip_sdf_optimization
from types import SimpleNamespace
import torch
import numpy as np
import skimage.measure
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import trimesh
import asyncio
import fastapi
import json
from fastapi import FastAPI, WebSocket
import math
app = fastapi.FastAPI()


polygon_worker = ThreadPoolExecutor(1)

class AsyncResult:
    def __init__(self):
        loop = asyncio.get_event_loop()
        self.signal = asyncio.Event(loop=loop)
        self.value = None

    def set(self, value):
        self.value = value
        self.signal.set()

    async def wait(self):
        await self.signal.wait()
        self.signal.clear()
        return self.value

    
async_result = None

@app.on_event("startup")
async def startup_event():
    global async_result
    async_result = AsyncResult()

class UserSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def run(self):
        await asyncio.gather(
            self.listen_loop(),
            self.send_loop()
        )

    async def listen_loop(self):
        while True:
            cmd = await self.websocket.receive_text()
            cmd = json.loads(cmd)
            print("XXX Got cmd", cmd)

    async def send_loop(self):
        while True:
            model = await async_result.wait()
            print(f"XXX WS sending model, {len(model)//1024}kb")
            await self.websocket.send_text(model)
            # await asyncio.sleep(1)
            # await self.websocket.send_text("HI")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    us = UserSession(websocket)
    await us.run()

import os
def process_sdf(sdf: np.ndarray):
    global async_result

    start = datetime.now()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    obj = trimesh.exchange.obj.export_obj(mesh)
    dur = datetime.now() - start
    if async_result:
        async_result.set(obj)
        print("XXX Posted message")
    else:
        print("XXX NO NOTIFIER???")
    print("Took", dur.total_seconds())

#%%
def on_update(mesh: torch.Tensor):
    print("XXX enqueuing preprocess")
    polygon_worker.submit(process_sdf, mesh.detach().cpu().numpy())

# sdf loss --> regulates that the shape is smooth
# lp loss --> regulates that the elements are altogether

import uvicorn

def run_sdf_clip():
    optim_config = SimpleNamespace(
        learning_rate=0.01,
        batch_size=1,
        init_tolerance=0.1,
        iters_per_res=6,
        max_iters_per_cam=32,
        camera=SimpleNamespace(
            max_num_cameras=64,
            init_num_cameras=16,
            mapping_span=2 * math.pi,
            shuffle_order=False,
            mapping_type="linear",
        ),
        loss=SimpleNamespace(
            image_loss_weight=1 / 1000,
            sdf_loss_weight=1 / 1000,
            lp_loss_weight=1 / 1000,
        ),
    )

    clip_sdf_optimization(
        prompt="3D bunny rabbit mesh rendered with maya zbrush",
        optim_config=optim_config,
        experiment_name="test",
        sdf_grid_res_list=[8, 16, 24, 32, 40, 48, 56, 64],
        on_update=on_update,
    )

def main():
    import threading
    thread = threading.Thread(target=run_sdf_clip, daemon=True)
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=9999, loop="asyncio")

if __name__ == "__main__":
    main()