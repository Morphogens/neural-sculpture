from sdf_clip import SDFCLIP
import torch
import numpy as np
import skimage.measure
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import trimesh
import asyncio
import fastapi
from fastapi import FastAPI, WebSocket
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
    print("XXX init!", os.getpid())
    async_result = AsyncResult()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        model = await async_result.wait()
        await websocket.send_text(model)
        await asyncio.sleep(1)

import os
def process_sdf(sdf: np.ndarray):
    global async_result

    start = datetime.now()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    obj = trimesh.exchange.dae.export_collada(mesh)
    dur = datetime.now() - start
    if async_result:
        async_result.set(obj)
    else:
        print("XXX NO NOTIFIER???")
    print("Took", dur.total_seconds())

#%%
def on_update(mesh: torch.Tensor):
    print("XXX HI!!!")
    polygon_worker.submit(process_sdf, mesh.detach().cpu().numpy())

# sdf loss --> regulates that the shape is smooth
# lp loss --> regulates that the elements are altogether

import uvicorn
def main():
    sdf_clip = SDFCLIP(
        prompt="Mesh of a bunny rabbit rendered with zbrush maya",
        on_update=on_update,
        out_img_width=256,
        out_img_height=256,
    )

    import threading
    thread = threading.Thread(target=sdf_clip.run, daemon=True)
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=9999, loop="asyncio")

if __name__ == "__main__":
    main()