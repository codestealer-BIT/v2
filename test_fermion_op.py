import torch
import io
import numpy as np
from PIL import Image
from PIL import ImageFile
from PIL import ImageOps
import requests

from IPython import embed
import fermion_ops
    
    
def _load_img_from_byte(byte_img, key):
    try:
        img = Image.open(io.BytesIO(byte_img)).convert('RGB')
        img = ImageOps.exif_transpose(img)
    except Exception:
        try:
            img = Image.open(io.BytesIO(byte_img))
            img.load()
        except Exception:
            pass
        try:
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)
        except Exception:
            print(f"Broken Imgae Loaded : {key}")
            img = None
    
    return img
    
    
def test_image_decode(image_byte):
    decode = fermion_ops.ImageDecodeOp(
        use_libjpeg_turbo=False, 
        resize_width=192, 
        resize_height=384,
        # resize_keep_aspect_ratio_method=fermion_ops.KeepAspectRatioMethod.SCALE_CROP,
    )    
    mean_pixel = torch.FloatTensor([0.48145466, 0.4578275, 0.40821073])
    scale = torch.FloatTensor([0.26862954, 0.26130258, 0.27577711])
    img1 = decode((image_byte,), is_video=False)[0]
    transpose = [0, 3, 1, 2]
    
    pil_image = Image.fromarray(img1[0].numpy().astype(np.uint8))
    pil_image.save("demo_output.jpg")
    img1 = img1.to(torch.float)
    img1 = torch.div(img1, 255)
    img1 = torch.sub(img1, mean_pixel)
    img1 = torch.div(img1, scale)
    img1 = img1.permute(transpose)

    # embed()
    

if __name__ == "__main__":
    with open("demo (3).jpg", 'rb') as f:
        img = f.read()
    test_image_decode(img)