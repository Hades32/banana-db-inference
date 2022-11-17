import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import base64
from io import BytesIO
import os
import re
import time
import zipfile

from minio import Minio
from minio.error import S3Error

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    print("init done")
    return

def dummy_safety_checker(images, clip_input):
    return images, false

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    # Parse out your arguments
    s3bucket = model_inputs.get("S3_BUCKET", os.getenv("S3_BUCKET"))
    s3client = Minio(
        model_inputs.get("S3_ENDPOINT", os.getenv("S3_ENDPOINT")),
        access_key=model_inputs.get("S3_KEY", os.getenv("S3_KEY")),
        secret_key=model_inputs.get("S3_SECRET", os.getenv("S3_SECRET")),
        region=model_inputs.get("S3_REGION", os.getenv("S3_REGION"))
    )
    single_prompt = model_inputs.get('prompt', None)
    prompts = model_inputs.get('prompts', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed", None)
    input_id = model_inputs.get("id", None)

    if input_id is None:
        return {"error": "missing input_id"}
    if single_prompt is None and prompts is None:
        return {'message': "No prompt provided"}
    if prompts is None:
        prompts = [single_prompt]

    downloadStart = time.monotonic_ns()
    os.makedirs("dreambooth_weights/", exist_ok=True)
    weightsObj = f'weights/{input_id}.zip'
    print(f"downloading {weightsObj}")
    s3client.fget_object(s3bucket, weightsObj, 'weights.zip')
    print(f"finished downloading in {(time.monotonic_ns() - downloadStart)/1_000_000_000}s")

    print("extracting weights")
    with zipfile.ZipFile('weights.zip', 'r') as f:
        f.extractall('dreambooth_weights')

    # workaround because I was stupid
    os.system("test -d dreambooth_weights/1200 && mv dreambooth_weights/1200 tmpmoveme && rm -rf dreambooth_weights && mv tmpmoveme dreambooth_weights")

    print("setting up pipeline")
    model = StableDiffusionPipeline.from_pretrained(
        "dreambooth_weights/", use_auth_token=HF_AUTH_TOKEN, safety_checker=dummy_safety_checker).to("cuda")

    # If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)

    print("Running the model")
    images = []
    with autocast("cuda"):
        for prompt in prompts:
            inferStart = time.monotonic_ns()
            image = model(prompt,
                          height=height, width=width,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          generator=generator).images[0]
            print(f"model ran in {(time.monotonic_ns() - inferStart)/1_000_000_000}s")
            images.append(image)

    image_paths = []
    for i, image in enumerate(images):
        bufferedImg = BytesIO()
        image.save(bufferedImg, format="JPEG")
        imgBytes = bufferedImg.getvalue()
        bufferedImg.seek(0)

        uploadStart = time.monotonic_ns()
        imgBucketFile = f"results/{input_id}/{re.sub(r'[^a-z -]+', ' ', prompts[i].lower())}_{uploadStart}.jpg"
        print(f"uploading {imgBucketFile}")
        s3client.put_object(s3bucket, imgBucketFile, bufferedImg, len(imgBytes))
        print(f"finished uploading in {(time.monotonic_ns() - uploadStart)/1_000_000_000}s")
        image_paths.append(imgBucketFile)

    # Return the results as a dictionary
    return {'image_paths': image_paths}
