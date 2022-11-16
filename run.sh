#!/bin/bash

# maybe make helper website from this

start=$(date)

callID=$(curl -s -XPOST 'https://api.banana.dev/start/v4/' -H 'Content-Type: application/json' -d '
{
    "startOnly": true,
    "apiKey": "'$API_KEY'",
    "modelKey": "'$MODEL_KEY'",
    "modelInputs": {
        "S3_ENDPOINT": "'$S3_ENDPOINT'",
        "S3_BUCKET": "'$S3_BUCKET'",
        "S3_KEY": "'$S3_KEY'",
        "S3_SECRET": "'$S3_SECRET'",
        "S3_REGION": "'$S3_REGION'",
        "id": "input-1_534295352088640",
        "prompt":  "a picture of a sks person sitting on a throne",
        "height":  512,
        "width":  512,
        "num_inference_steps":  50,
        "guidance_scale":  8.5,
        "seed": 42
    }
}' | jq -j .callID)

echo "started ${callID} at ${start}"

while curl -s -XPOST 'https://api.banana.dev/check/v4/' -H 'Content-Type: application/json' -d'
{    
    "apiKey": "'$API_KEY'",    
    "callID": "'$callID'"
}' | grep running > /dev/null ; do 
    echo -n .
done

echo
curl -s -XPOST 'https://api.banana.dev/check/v4/' -H 'Content-Type: application/json' -d'
{    
    "apiKey": "'$API_KEY'",    
    "callID": "'$callID'"
}' | jq .
echo "$start until $(date)"
