import time
import os
from typing import Sized
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from network.Transformer import Transformer


def transform(models, style, input, load_size=450, gpu=-1):
    model = models[style]

    if gpu > -1:
        model.cuda()
    else:
        model.float()

    input_image = Image.open(input).convert("RGB")
    h, w = input_image.size
    ratio = h * 1.0 / w

    if ratio > 1:
        h = load_size
        w = int(h * 1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)

    # resizing the image
    # input_image=input_image.resize(size=(h, w,),Image.BICUBIC)
    input_image = input_image.resize((h, w), Image.BICUBIC)

    input_image = np.asarray(input_image)
    # converting image from RCB to BGR
    input_image = input_image[:, :, [2, 1, 0]]

    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = -1 + 2 * input_image  # normalising as specified in Paper

    if gpu > -1:
        input_image = Variable(input_image).cuda()
    else:
        input_image = Variable(input_image).float()

    # Transform Input image using Style Transfer
    t0 = time.time()
    with torch.no_grad():
        output_image = model(input_image)[0]
    print(f"{style} - Inference time taken: {time.time()-t0} s")

    # Tranfomring output image from BGR to RGB
    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
    output_image = output_image.numpy()
    output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)
    output_image = Image.fromarray(output_image)
    return output_image
