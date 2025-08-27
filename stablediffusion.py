'''import boto3
import json
import base64
import os


prompt="""

provide me one 4k hd image of a man swimming in the river.

"""

prompt_template = {"text": prompt, "weight": 1}


bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    
  "text_prompts": prompt_template,
  "cfg_scale": 10,
  "seed": 0,
  "steps": 50,
  "width": 512,
  "height": 512


    
}
body=json.dumps(payload)



model_id = "stability.stable-diffusion-xl-v1"





response = bedrock.invoke_model(
    body=body,
    modelId= model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.loads(response.get["body"].read())
print(response_body)

#resnse_body converted to an image and stores in the current directory

artifacts = response_body.get("artifacts")[0]
image_encoded = artifacts.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)


output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)
'''


import boto3
import json
import base64

# ---- Step 1: Configure AWS Bedrock Runtime client ----
# Make sure your AWS credentials and region are correctly set in your environment
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"  # Change to "us-west-2" if that's your Bedrock region
)

# ---- Step 2: Define the correct Stable Diffusion model ID ----
# Official Bedrock model IDs:
# - stability.stable-diffusion-xl-v1
# - stability.stable-diffusion-xl-v0
model_id = "stability.stable-diffusion-xl-v1"

# ---- Step 3: Define your text-to-image prompt ----
prompt = {
    "text_prompts": [
        {"text": "A futuristic Lowe’s smart store with AI-powered checkout counters, high resolution, 8k render"}
    ],
    "cfg_scale": 8.0,   # Creativity vs accuracy (higher = closer to prompt)
    "steps": 50,        # Number of diffusion steps
    "seed": 12345       # Random seed for reproducibility
}

# ---- Step 4: Call Bedrock InvokeModel API ----
try:
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(prompt),
        contentType="application/json",
        accept="application/json"
    )

    # ---- Step 5: Parse response ----
    result = json.loads(response["body"].read())
    image_base64 = result["artifacts"][0]["base64"]

    # ---- Step 6: Save image to file ----
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(image_base64))

    print("✅ Image successfully generated and saved as output.png")

except Exception as e:
    print(f"❌ Error generating image: {e}")


