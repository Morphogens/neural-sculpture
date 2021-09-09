from pydantic import BaseModel
import math

class Loss(BaseModel):
    image_loss_weight: float =1 / 1000
    sdf_loss_weight: float =1 / 1000
    lp_loss_weight: float =1 / 1000

class Camera(BaseModel):
    max_num_cameras: int=16
    init_num_cameras: int=8
    mapping_span: float=2*math.pi
    mapping_offset=math.pi
    cam_scaler = 7.5
    shuffle_order: bool=False
    mapping_type: str = "linear"

class OptimConfig(BaseModel):
    learning_rate: float =0.01
    batch_size: int =1
    init_tolerance: float =-0.1
    iters_per_res: int=10
    max_iters_per_cam: int = 8
    loss: Loss = Loss()
    camera: Camera = Camera()
