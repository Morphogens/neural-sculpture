import os
import math
import json
import threading
from typing import *
from types import SimpleNamespace
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
sdf_grid_res_list = [16, 24, 40, 64]


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


def generate_from_coord(coord: List[int], ):
    print("CURSOR TXT", coord)

    sdf_optimizer.optimize_coord(
        coord=coord,
        prompt="3D bunny rabbit gray mesh rendered with maya zbrush",
    )


class UserSession:
    def __init__(
        self,
        websocket: WebSocket,
    ):
        self.websocket = websocket

    async def run(self, ):
        await asyncio.gather(self.listen_loop(), self.send_loop())

    async def listen_loop(self, ):
        while True:
            cmd = await self.websocket.receive_text()
            cmd = json.loads(cmd)

            print("XXX Got cmd", cmd)

            if cmd['message'] == 'cursor':
                data_dict = cmd['data']
                if data_dict is not None:
                    coord = data_dict['point']
                    generate_from_coord(coord)

    async def send_loop(self, ):
        while True:
            model = await async_result.wait()
            print(f"XXX WS sending model, {len(model)//1024}kb")
            await self.websocket.send_text(model)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    us = UserSession(websocket)
    await us.run()


def process_sdf(sdf: np.ndarray):
    global async_result

    start = datetime.now()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           vertex_normals=normals)
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

sdf_optimizer = reset_sdf_optimizer()

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