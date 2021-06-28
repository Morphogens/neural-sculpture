import clip
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from PIL import Image

device = "cuda"
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_res = 224
clip_norm = Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)
def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x

    return TF.to_tensor(x)


# slightly modified from OpenAI's code, so that it works with np tensors
# see https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/clip.py#L58
clip_preprocess = Compose(
    [
        to_tensor,
        Resize(clip_res, interpolation=Image.BICUBIC),
        CenterCrop(clip_res),
        clip_norm,
    ]
)

class CLIPLoss(torch.nn.Module):
    def __init__(self, text_target):
        super().__init__()
        self.text = clip.tokenize([text_target]).to(device)

        # self.upsample = torch.nn.Upsample(scale_factor=7)
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=imageSize // 32)
        # self.text = torch.cat([clip.tokenize(textTarget)]).cuda()

    # def forward(self, image):
        # image = self.avg_pool(self.upsample(image))
        # similarity = 1 - self.model(image, self.text)[0] / 100
        # return similarity    

    def forward(self, img):
        N, C, H, W = img.shape
        # assert H == W, f"images should be rectangular but was {W}x{H}"
        # img = torch.nn.functional.interpolate(img, (224, 224))
        x = clip_preprocess(img)
        img_logits = clip_model(x, self.text)[0]
        # loss = 1/img_logits * 100
        loss = 1-img_logits/100
        return loss.float()
        # loss = 1 - sim_score / 100
        # return loss.float() * 1000