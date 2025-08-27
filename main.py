import os
import json
import sys
import boto3

print("Imported successfully....")

# Mentioning the prompt here

prompt = """

    You are a helpful assistant so please let me know what is machine learning in a smartest way?

"""
# assiginng the below to a variable

bedrock=boto3.client(service_name= "bedrock-runtime")





payload = {
    ...
}
body = json.dumps(payload)
model_id="Llama 3.1 70B Instruct"

#meta.llama2-70b-chat-v1

response=bedrock.invoke_model(
    body=body,
    model_id=model_id,
    accept="application/json",
    contentType="application/json"
    
)

response_body=json.loads(response["body"].read())
response_text=response_body["generation"]
print(response_text)


