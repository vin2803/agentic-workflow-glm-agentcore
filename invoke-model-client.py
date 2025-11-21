# python code to invoke endpoint, payload subject to model

import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")
#runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

payload = {
        "inputs": "What is machine learning?"
}

LLAMA_LONGER_INPUT_JSON = {
            "model": "/opt/ml/model",
            "messages": [
                {
                    "role": "user",
                    "content": "Where should I visit then?"
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.0
        }

LLAMA_LONGER_INPUT_JSON_1 = {
      "messages": [
            {"role": "user", "content": "How many times does the letter b appear in blueberry?"}
        ],
        "chat_template_kwargs": {
            "enable_thinking": False
        }
}

response = runtime.invoke_endpoint(
    #EndpointName="deepseek15b-20250725130449",
    EndpointName="glm4-5-2025-11-05-23-19-04-348",
    #InferenceComponentName="ic-gpt-oss-120b-2025-11-05-01-53-27-686",
    ContentType="application/json",
    #TargetModel="deepseek15b20250725130449-default-Bwnq6uPr",
    Body=json.dumps(LLAMA_LONGER_INPUT_JSON_1)
)

print(response["Body"].read().decode())