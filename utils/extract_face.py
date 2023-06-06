import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from PIL import Image
import cv2

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size

def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data

def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out

def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)


    face = F.to_tensor(np.float32(face))

    return face

def extract(img, batch_boxes, margin=0, post_process=True, keep_all=False):
    # Determine if a batch or single image was passed
    batch_mode = True
    if (
            not isinstance(img, (list, tuple)) and
            not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
            not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
    ):
        img = [img]
        batch_boxes = [batch_boxes]
        batch_mode = False

    # Parse save path(s)

    save_path = [None for _ in range(len(img))]

    # Process all bounding boxes
    faces = []
    for im, box_im, path_im in zip(img, batch_boxes, save_path):
        if box_im is None:
            faces.append(None)
            continue

        faces_im = []
        for i, box in enumerate(box_im):
            face_path = path_im

            face = extract_face(im, box, 160, margin, face_path)
            if post_process:
                face = fixed_image_standardization(face)
            faces_im.append(face)

        if keep_all:
            faces_im = torch.stack(faces_im)
        else:
            faces_im = faces_im[0]

        faces.append(faces_im)

    if not batch_mode:
        faces = faces[0]

    return faces