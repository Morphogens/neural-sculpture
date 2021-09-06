import os
import math
import json
import threading
from typing import *
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

from clip_sdf import SDFOptimizer, CLIPSDFConfig

app = fastapi.FastAPI()

polygon_worker = ThreadPoolExecutor(1)

# XXX: MAYBE BEST PARAMS EVER!!
optim_config = CLIPSDFConfig(
    learning_rate=0.01,
    batch_size=1,
    init_tolerance=-0.2,
    iters_per_res=6,
    max_iters_per_cam=4,
    camera=CLIPSDFConfig(
        max_num_cameras=16,
        init_num_cameras=8,
        mapping_span=math.pi,
        mapping_offset=math.pi,
        shuffle_order=False,
        mapping_type="linear",
    ),
    loss=CLIPSDFConfig(
        image_loss_weight=1 / 1000,
        sdf_loss_weight=0 / 1000,
        lp_loss_weight=0 / 1000,
    ),
)
# sdf_grid_res_list = [16, 24, 40, 64]


def reset_sdf_optimizer(
    sdf_grid_res_list=[64],
    sdf_dir="./sdf-grids",
    sdf_filename="cat.npy",
):
    sdf_optimizer = SDFOptimizer(
        config=optim_config,
        sdf_grid_res_list=sdf_grid_res_list,
        sdf_file_path=os.path.join(sdf_dir, sdf_filename),
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


# def generate_from_coord(coord: List[int], ):
#     print("CURSOR TXT", coord)

#     sdf_optimizer.optimize_coord(
#         coord=coord,
#         prompt="3D bunny rabbit gray mesh rendered with maya zbrush",
#     )


class UserSession:
    def __init__(
        self,
        websocket: WebSocket,
    ):
        self.websocket = websocket

    async def run(self, ):
        await asyncio.gather(self.listen_loop(), self.send_loop())

    async def listen_loop(self, ):
        sdf_optimizer = reset_sdf_optimizer()
        while True:
            cmd = await self.websocket.receive_text()
            cmd = json.loads(cmd)

            print("XXX Got cmd", cmd)

            if cmd['message'] == 'initialize':
                sdf_filename = cmd['data']
                if 'npy' not in sdf_filename:
                    sdf_filename += ".npy"

                sdf_optimizer = reset_sdf_optimizer(
                    sdf_filename=sdf_filename, )

                polygon_worker.submit(
                    process_sdf,
                    sdf_optimizer.grid.detach().cpu().numpy(),
                )

            if cmd['message'] == 'cursor':
                data_dict = cmd['data']
                if data_dict is not None:
                    coord = data_dict['point']
                    # coord = list(np.moveaxis(np.array(coord)), 0, -1)

                    print("XXX OPTIMIZING COORD")
                    grid = sdf_optimizer.optimize_coord(
                        coord=coord,
                        prompt=
                        "3D bunny rabbit gray mesh rendered with maya zbrush",
                    )

                    process_sdf(grid.detach().cpu().numpy(), )

                    # polygon_worker.submit(
                    #     process_sdf,
                    #     grid.detach().cpu().numpy(),
                    # )

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
    print("XXX processing sdf")
    global async_result

    start = datetime.now()

    if sdf.max() <= 0 or sdf.min() >= 0:
        print("XXX SDF SHAPE EMPTY OR FULL OF NOISE")

    print("XXX marching cubes")
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)

    print("XXX mesh")
    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           vertex_normals=normals)

    print("XXX obj")
    obj = trimesh.exchange.obj.export_obj(mesh)

    print("XXX dur")
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
    sdf_optimizer = reset_sdf_optimizer()

    sdf_optimizer.clip_sdf_optimization(
        prompt="3D bunny rabbit gray mesh rendered with maya zbrush",
        experiment_name="test",
    )


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