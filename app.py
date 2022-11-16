import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import base64
from io import BytesIO
import os

from minio import Minio
from minio.error import S3Error

s3bucket = os.environ["S3_BUCKET"]
s3client = Minio(
    os.environ["S3_ENDPOINT"],
    access_key=os.environ["S3_KEY"],
    secret_key=os.environ["S3_SECRET"],
    region=os.environ["S3_REGION"],
)

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    if os.environ["S3_ENDPOINT"] == "":
        raise RuntimeError("S3_ENDPOINT not set")
    print("init done")
    return

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    # Parse out your arguments
    prompt = model_inputs.get('prompt', "a picture of a sks person")
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)
    input_id = model_inputs.get("id",None)

    if input_id is None:
        return {"error": "missing input_id"}

    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    os.makedirs("dreambooth_weights/", exist_ok=True)
    weightsObj = f'weights/{input_id}.zip'
    print(f"downloading {weightsObj}")
    s3client.fget_object(s3bucket, weightsObj, 'weights.zip')
    with zipfile.ZipFile('weights.zip', 'r') as f:
        f.extractall('dreambooth_weights')

    print("setting up pipeline")
    model = StableDiffusionPipeline.from_pretrained("dreambooth_weights/",use_auth_token=HF_AUTH_TOKEN).to("cuda")
    
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    print("Runing the model")
    with autocast("cuda"):
        image = model(prompt,height=height,width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator)["sample"][0]

    buffered = BytesIO()
    image.save(buffered,format="JPEG")

    print("uploading result")
    s3client.put_object(s3bucket, f"results/{input_id}/i1.jpg", buffered, buffered.getbuffer().nbytes)

    # Return the results as a dictionary
    return {'image_base64': base64.b64encode(buffered.getvalue()).decode('utf-8')}
