import base64
import json
import os
import re
import time
import zipfile
from io import BytesIO

import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from minio import Minio
from minio.error import S3Error
from torch import autocast

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    print("init done!")
    return

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
    input_model = model_inputs.get("weights_path", None)
    output_path = model_inputs.get("output_path", None)

    try:

        if input_model is None:
            raise Exception("missing input_model")
        if output_path is None:
            raise Exception("missing output_path")
        if single_prompt is None and prompts is None:
            raise Exception("No prompt provided")
        if prompts is None:
            prompts = [single_prompt]

        downloadStart = time.monotonic_ns()
        os.makedirs("dreambooth_weights/", exist_ok=True)
        print(f"downloading {input_model}")
        s3client.fget_object(s3bucket, input_model, 'weights.zip')
        print(
            f"finished downloading in {(time.monotonic_ns() - downloadStart)/1_000_000_000}s")

        print("extracting weights")
        with zipfile.ZipFile('weights.zip', 'r') as f:
            f.extractall('dreambooth_weights')

        # workaround because I was stupid
        os.system("test -d dreambooth_weights/1200 && mv dreambooth_weights/1200 tmpmoveme && rm -rf dreambooth_weights && mv tmpmoveme dreambooth_weights")

        print("setting up pipeline")
        model = StableDiffusionPipeline.from_pretrained(
            "dreambooth_weights/", use_auth_token=HF_AUTH_TOKEN, safety_checker=None).to("cuda")

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
                print(
                    f"model ran in {(time.monotonic_ns() - inferStart)/1_000_000_000}s")
                images.append(image)

        generations = []
        upload_id = int(time.time())
        for i, image in enumerate(images):
            bufferedImg = BytesIO()
            image.save(bufferedImg, format="JPEG")
            bufferedImg.seek(0)
            uploadStart = time.monotonic_ns()
            imgBucketFile = f"img_{upload_id}_{i}.jpg"
            print(f"uploading {imgBucketFile}")
            s3client.put_object(s3bucket, f"{output_path}/{imgBucketFile}",
                                bufferedImg, len(bufferedImg.getbuffer()))
            print(
                f"finished uploading in {(time.monotonic_ns() - uploadStart)/1_000_000_000}s")
            generations.append({'path': imgBucketFile, 'prompt': prompts[i]})

        # Return and save results
        result = {'generations': generations, 'finished_at': time.time()}
        json_data = BytesIO(json.dumps(result).encode())
        s3client.put_object(
            s3bucket, f"{output_path}/results.json", json_data, len(json_data.getbuffer()))

    except Exception as err:
        print("some exception occured:")
        print(err)
        result = {'error': err.__str__()}
        json_data = BytesIO(json.dumps(result).encode())
        s3client.put_object(
            s3bucket, f"{output_path}/results.json", json_data, len(json_data.getbuffer()))

    return result
