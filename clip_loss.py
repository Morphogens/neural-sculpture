import torch
import torchvision
import clip


class CLIPLoss:
    def __init__(
        self,
        text_target=None,
        device=None,
    ):
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"ClipLoss device {self.device}")

        self.clip_input_img_size = 224

        self.clip_model, _clip_preprocess = clip.load(
            "ViT-B/32",
            device=self.device,
        )
        self.clip_model = self.clip_model.eval()

        self.clip_norm_trans = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )

        self.aug_transform = torch.nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(24, (.05, .05)),
        ).to(self.device)

        if text_target is not None:
            tokenized_text = clip.tokenize([text_target])
            tokenized_text = tokenized_text.to(self.device).detach()
            text_logits = self.clip_model.encode_text(tokenized_text)
            self.text_logits = (
                text_logits / text_logits.norm(dim=-1, keepdim=True)).detach()

    def augment(
        self,
        img_batch,
        num_crops=32,
    ):
        target_img_size = img_batch.shape[-1]
        # print('img_batch.shape', img_batch.shape)
        # assert target_img_size > 8, target_img_size

        pad_size = target_img_size // 2
        img_batch = torch.nn.functional.pad(
            img_batch,
            (
                pad_size,
                pad_size,
                pad_size,
                pad_size,
            ),
            mode='constant',
            value=0,
        )

        img_batch = self.aug_transform(img_batch)

        augmented_img_list = []
        for crop in range(num_crops):
            crop_size = int(
                torch.normal(
                    1.2,
                    .3,
                    (),
                ).clip(.7, 1.3) * target_img_size)

            if crop > num_crops - 4:
                crop_size = int(target_img_size * 1.4)

            offsetx = torch.randint(
                0,
                int(target_img_size * 2 - crop_size),
                (),
            )
            offsety = torch.randint(
                0,
                int(target_img_size * 2 - crop_size),
                (),
            )
            augmented_img = img_batch[:, :, offsetx:offsetx + crop_size,
                                      offsety:offsety + crop_size, ]
            augmented_img = torch.nn.functional.interpolate(
                augmented_img,
                (224, 224),
                mode='bilinear',
                align_corners=True,
            )
            augmented_img_list.append(augmented_img)

        img_batch = torch.cat(augmented_img_list, 0)

        up_noise = 0.11
        img_batch = img_batch + up_noise * torch.rand(
            (img_batch.shape[0], 1, 1, 1)).to(self.device) * torch.randn_like(
                img_batch, requires_grad=False)

        img_batch = img_batch.clamp(0, 1)

        return img_batch

    def get_clip_img_encodings(
        self,
        img_batch: torch.Tensor,
        do_preprocess: bool = True,
    ):
        if do_preprocess:
            img_batch = self.clip_norm_trans(img_batch)
            img_batch = torch.nn.functional.upsample_bilinear(
                img_batch,
                (self.clip_input_img_size, self.clip_input_img_size),
            )

        img_logits = self.clip_model.encode_image(img_batch)
        img_logits = (img_logits / img_logits.norm(dim=-1, keepdim=True))

        return img_logits

    def compute(
        self,
        img_batch: torch.Tensor,
        prompt: str = None,
        augment: bool = True,
    ):
        if augment:
            img_batch = self.augment(img_batch)
        img_logits = self.get_clip_img_encodings(img_batch)

        if prompt is not None:
            tokenized_text = clip.tokenize([prompt])
            tokenized_text = tokenized_text.to(self.device).detach()
            text_logits = self.clip_model.encode_text(tokenized_text)
            text_logits = (text_logits /
                           text_logits.norm(dim=-1, keepdim=True)).detach()
        else:
            text_logits = self.text_logits

        loss = 0
        # loss += (text_logits - img_logits).pow(2).mean()
        # loss += (text_logits -
        #          img_logits).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()
        loss = 1-torch.cosine_similarity(text_logits, img_logits).mean()

        return loss, img_logits


# class ClipLoss(ImageGenerator):
#     def __init__(self, text_target):

# img_aug = augment_trans(img)
# image_features = model.encode_image(img_aug)
# loss -= torch.cosine_similarity(features, image_features, dim=1)/8.0
