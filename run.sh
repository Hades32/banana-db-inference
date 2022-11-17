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
        "prompts":  [
            "character study of a sks person, clear faces, wild, crazy, character sheet, fine details, concept design, contrast, kim jung gi, pixar and da vinci, trending on artstation, 8 k, full body and head, turnaround, front view, back view, ultra wide angle ",
            "roman bust of a sks person",
            "photo of a classical greek marble statue of a sks person, [ ultra detail, intricate, masterpiece, dslr, museum ]",
            "photo of Michelangelo`s sks person statue, [ ultra detail, intricate, masterpiece, dslr, museum ]"
        ],
        "height":  512,
        "width":  512,
        "num_inference_steps":  80,
        "guidance_scale":  8.5,
        "seed": 42424242424242
    }
}' | jq -j .callID)

echo "started ${callID} at ${start}"

while export result=$(curl -s -XPOST 'https://api.banana.dev/check/v4/' -H 'Content-Type: application/json' -d'
{    
    "apiKey": "'$API_KEY'",    
    "callID": "'$callID'"
}') && echo "$result" | grep running > /dev/null ; do 
    echo -n .
done

echo "$result" | jq .
echo "$start until $(date)"
