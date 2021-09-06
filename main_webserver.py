import os
import math
import json
import threading
from typing import *
from types import SimpleNamespace
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import torch
import asyncio
import trimesh
import fastapi
import uvicorn
import skimage.measure
import numpy as np
from fastapi import FastAPI, WebSocket

from clip_sdf import SDFOptimizer

app = fastapi.FastAPI()

polygon_worker = ThreadPoolExecutor(1)
inferece_worker = ThreadPoolExecutor(1)

# XXX: MAYBE BEST PARAMS EVER!!
optim_config = SimpleNamespace(
    learning_rate=0.003,
    batch_size=1,
    init_tolerance=-0.2,
    iters_per_res=6,
    max_iters_per_cam=4,
    camera=SimpleNamespace(
        max_num_cameras=16,
        init_num_cameras=8,
        mapping_span=math.pi,
        mapping_offset=math.pi,
        shuffle_order=False,
        mapping_type="linear",
    ),
    loss=SimpleNamespace(
        image_loss_weight=1 / 1000,
        sdf_loss_weight=0 / 1000,
        lp_loss_weight=0 / 1000,
    ),
)
sdf_grid_res_list = [64]


def reset_sdf_optimizer():
    sdf_optimizer = SDFOptimizer(
        config=optim_config,
        sdf_grid_res_list=sdf_grid_res_list,
        sdf_file_path=None, # "./sdf-grids/cat-sdf.npy",
        on_update=on_update,
    )

    return sdf_optimizer




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
    def __init__(
        self,
        websocket: WebSocket,
    ):
        self.websocket = websocket
        self.coord = None
        self.prompt = "3D bunny rabbit gray mesh rendered with maya zbrush"
        self.run_tick = True

    async def run(self):
        await asyncio.wait([self.listen_loop(), self.send_loop()], return_when=asyncio.FIRST_COMPLETED)

    async def listen_loop(self):
        while True:
            cmd = await self.websocket.receive_text()
            cmd = json.loads(cmd)

            if cmd['message'] == 'cursor':
                data_dict = cmd['data']
                if data_dict is not None:
                    self.coord = data_dict['point']
                    print("cursor is at", self.coord)

    async def send_loop(self):
        while True:
            model = await async_result.wait()
            print(f"XXX WS sending model, {len(model)//1024}kb")
            await self.websocket.send_text(model)


class OptimizerWorker:
    user_session: Optional[UserSession]
    def __init__(self):
        self.user_session = None
        self.optimizer_thread = Thread(target=self.optimizer_loop, daemon=True)
        self._running = True
        self.optimizer_thread.start()

    def optimizer_loop(self):
        sdf_optimizer = reset_sdf_optimizer()
        while self._running:
            us = self.user_session
            if not us or not us.run_tick:
                print("No work...")
                time.sleep(1/30)
                continue
            
            if us.coord is not None:
                print(f"running optimizer with prompt {us.prompt}")
                sdf_optimizer.optimize_coord(
                    coord=us.coord,
                    prompt=us.prompt
                )

            # HACK: Test out all code
            # sdf_optimizer.optimize_coord(self.prompt)
        print("XXX optimizer stopped")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    us = UserSession(websocket)
    optimization_worker.user_session = us
    await us.run()

def process_sdf(sdf: np.ndarray, loss_dict: Dict[str, float]):
    print("loss", " ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items()))
    start = datetime.now()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           vertex_normals=normals)
    obj = trimesh.exchange.obj.export_obj(mesh)
    dur = datetime.now() - start

    result = dict(
        obj=obj,
        **loss_dict
    )

    if async_result:
        async_result.set(json.dumps(result))
        print("XXX Posted message")
    else:
        print("XXX NO NOTIFIER???")
    print("Took", dur.total_seconds())


#%%
def on_update(*args, **kwargs):
    print("XXX enqueuing preprocess")
    polygon_worker.submit(process_sdf, *args, **kwargs)


# sdf loss --> regulates that the shape is smooth
# lp loss --> regulates that the elements are altogether


def run_sdf_clip():
    sdf_optimizer = SDFOptimizer(
        config=optim_config,
        sdf_grid_res_list=sdf_grid_res_list,
        sdf_file_path="./sdf-grids/cat-sdf.npy",
        on_update=on_update,
    )

    sdf_optimizer.clip_sdf_optimization(
        prompt="3D bunny rabbit gray mesh rendered with maya zbrush",
        experiment_name="test",
    )

optimization_worker = OptimizerWorker()

def main():
    # thread = threading.Thread(
    #     target=run_sdf_clip,
    #     daemon=True,
    # )
    # thread.start()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        loop="asyncio",
    )



if __name__ == "__main__":
    main()