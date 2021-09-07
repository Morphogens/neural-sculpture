import os
import math
from typing import *
from datetime import datetime

import torch
import trimesh
import skimage.measure
import numpy as np

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from clip_sdf import SDFOptimizer, CLIPSDFConfig

app = Flask(__name__)
CORS(app)

# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(
    app,
    cors_allowed_origins='*',
)

app.app_context().push()


# app = fastapi.FastAPI()
@socketio.on('initialize')
def initialize_mesh(
    self,
    args_dict,
):
    print('received args: ', args_dict)


class SDFServer:
    def __init__(self, ):
        self.sdf_grid_res_list = [16, 24, 40, 64]

        self.sdf_file_path = None

        self.optim_config = CLIPSDFConfig(
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

        self.initialize_sdf_optimizer()

    def initialize_sdf_optimizer(self, ):
        self.sdf_optimizer = SDFOptimizer(
            config=self.optim_config,
            sdf_grid_res_list=self.sdf_grid_res_list,
            sdf_file_path=self.sdf_file_path,
            on_update=self.send_sdf_grid,
        )

    def set_sdf_file_path(
        self,
        sdf_filename: str,
        sdf_dir: str = "./sdf-grids",
    ):
        self.sdf_file_path = os.path.join(
            sdf_dir,
            sdf_filename,
        )

        return self.sdf_file_path

    @socketio.on('message')
    def handle_message(data):
        print('received message: ' + data)

    @socketio.on('initialize')
    def initialize_mesh(
        self,
        args_dict,
    ):
        print('received args: ', args_dict)

        sdf_filename = args_dict['sdfFilename']
        if 'npy' not in sdf_filename:
            sdf_filename += ".npy"

        self.set_sdf_file_path(sdf_filename=sdf_filename)
        self.initialize_sdf_optimizer()

        sdf_grid = self.sdf_optimizer.grid.detach().cpu().numpy(),
        _ = self.send_sdf_grid(sdf_grid)

    @socketio.on('runOptimization')
    def run_optimization(
        self,
        _args_dict,
    ):
        print("XXX OPTIMIZING...")
        sdf_grid = self.sdf_optimizer.clip_sdf_optimization(
            prompt="3D bunny rabbit gray mesh rendered with maya zbrush", )

        self.send_sdf_grid(sdf_grid)

    @socketio.on('optimizeCoord')
    def optimize_coord(
        self,
        args_dict,
    ):
        coord = args_dict['coord']

        sdf_grid = self.sdf_optimizer.optimize_coord(
            coord=coord,
            prompt="3D bunny rabbit gray mesh rendered with maya zbrush",
        )

        self.send_sdf_grid(sdf_grid, )

    @staticmethod
    def send_sdf_grid(sdf_grid: Union[np.ndarray, torch.Tensor], ):
        print("XXX processing sdf")

        start = datetime.now()

        if type(sdf_grid) == torch.Tensor:
            sdf_grid = sdf_grid.detach().cpu().numpy()

        if sdf_grid.max() <= 0 or sdf_grid.min() >= 0:
            print("XXX SDF SHAPE EMPTY OR FULL OF NOISE")

        vertices, faces, normals, _ = skimage.measure.marching_cubes(
            sdf_grid,
            level=0,
        )

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
        )

        mesh_obj = trimesh.exchange.obj.export_obj(mesh, )

        dur = datetime.now() - start

        print("Took", dur.total_seconds())

        response = jsonify(
            success=True,
            results=mesh_obj,
        )

        emit('mesh', response.json)

        return


if __name__ == "__main__":
    socketio.run(
        app=app,
        # host="0.0.0.0",
        port=8005,
    )
