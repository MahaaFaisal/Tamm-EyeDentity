import json
import base64
from openai import AzureOpenAI
from PIL import Image
import os


def encode_image_to_base64(self, image_path: str) -> str:
		"""Encode image to base64."""
		with open(image_path, 'rb') as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

def qaqc_agent(image_path, predictions):
    endpoint = "https://api.core42.ai/"
    model_name = "gpt-4o"
    deployment = "gpt-4o"

    subscription_key = "API_KEY"
    api_version = "2024-08-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    image_bytes = None
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    response = client.chat.completions.create(
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant designed to output JSON."
            },
            {
                "role": "user", 
                 "content": [
	                    {
	                        "type": "text",
	                        "text": """
                                    You Are a Quality Control Agent. Your job is to assess the quality of damage-assessment predictions.
                                    [TASK]
                                    You will receive a photo of a one damaged car no need to look for other cars it is only one car at the image. along with three separate damage-assessment predictions.
                                        1. Examine the damaged car image closely.
                                        2. Compare the three predictions.
                                        3. Choose the prediction that most accurately matches the actual damage in the photo.
                                        4. If any real, visible damage appears in one prediction but is missing from the best overall prediction, merge that missing detail into your final answer.
                                        5. if any prediction is wrong, you should ignore it.
                                        6. after make sure that the final prediction is in the same categories as the original predictions.
                                    [/TASK]
                                    [INPUT_EXAMPLE]
                                        ["Car Image With Flat tire and Scratch"], ["Car Image With Crack"], ["Car Image With Flat tire and Scratch and Broken lamp"]}
                                    [/INPUT_EXAMPLE]
                                    [OUTPUT_EXAMPLE]
                                        {"prediction": "Car Image With Flat tire and Scratch and broken lamp."}
                                    [/OUTPUT_EXAMPLE]
                                    [RESPONSE_EXAMPLE]
                                        Your final answer should be a single JSON object with the following format:
                                            {"prediction": "Your prediction here"}
                                    [/RESPONSE_EXAMPLE]
                                    """
	                    },
	                    {
	                        "type": "image_url",
	                        "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""{{"predictions": {predictions}}}"""
                        },
                        {
                            "type": "text",
                            "text": """
                                    [PREDICTION CATEGORIES]
                                        1. Dent.
                                        2. Scratch.
                                        3. Crack.
                                        4. Shattered glass.
                                        5. Broken lamp.
                                        6. Flat tire.
                                    [/PREDICTION CATEGORIES]
                                    """
                        }
                    ] 
            },
            {
                "role": "user",
                "content": f"""Response should be in JSON format. with the
                            following {{"prediction" : "Your prediction here"}} """,
            }
            ],
            temperature=0.0,
            model=deployment
        )
    response_json = response.choices[0].message.content
    # Decode the JSON response
    try:
        response_json = json.loads(response_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None
    # Check if the response contains the expected keys
    if 'prediction' not in response_json:
        print("Error: 'prediction' key not found in the response.")
        return None
    cleaned_prediction = response_json['prediction'].replace('"', '')
    cleaned_prediction = cleaned_prediction.replace(',', ' and')
    return cleaned_prediction

def assessor_agent(image_path, temperatures):
    endpoint = "https://api.core42.ai/"
    model_name = "gpt-4o"
    deployment = "gpt-4o"

    subscription_key = "API_KEY"
    api_version = "2024-08-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    image_bytes = None
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    response = client.chat.completions.create(
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant designed to output JSON."
            },
            {
                "role": "user", 
                 "content": [
	                    {
	                        "type": "text",
	                        "text": """
                                        [TASK]
                                            You are a professional Vehicle Damage Assessor. Your job is to accurately identify visible damage on a car based on the provided image.
                                            the image that you see has only one car with one or multiple damage categories.
                                            [STEPS]
                                                STEP 1 - Focus the Vehicle  
                                                    • Locate the only car in the image (ignore the background).
                                                STEP 2 - Inspect for Damage  
                                                    • Sweep the body, windows, lamps, and tires for any of the six damage types.  
                                                    • A vehicle may show one, several.
                                                STEP 3 -  Build the Result String  
                                                    • If damage exists, list **each** detected category once, separated by and.  
                                                        Example: `Dent and Scratch and Broken Lamp`  
                                                STEP 4 - Output JSON  
                                                    • Return nothing but a single JSON object:  
                                                        `{"prediction": "Car Image With Dent and Scratch and Broken Lamp."}`
                                            [/STEPS]
                                            [DAMAGE CATEGORIES]
                                                1. Dent.
                                                2. Scratch.
                                                3. Crack.
                                                4. Shattered glass.
                                                5. Broken lamp.
                                                6. Flat tire.
                                            [/DAMAGE CATEGORIES]
                                        [/TASK]
                                        [RESPONSE_EXAMPLE]
                                            Example 1 - {"prediction": "Car Image With Flat tire and Scratch"}  
                                            Example 2 - {"prediction": "Car Image With Crack."}  
                                            Example 3 - {"prediction": "Car Image With Dent and Shattered glass and Broken lamp."}  
                                            Example 4 - {"prediction": "Car Image With No visible damage detected."}
                                        [/RESPONSE_EXAMPLE]
                                        """
	                    },
	                    {
	                        "type": "image_url",
	                        "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        } 
                    ] 
            },
            {
                "role": "user",
                "content": f"""Response should be in JSON format. with the
                            following {{"prediction" : "Your prediction here"}} """,
            }
            ],
            temperature=temperatures,
            model=deployment
        )
    response_json = response.choices[0].message.content
    # Decode the JSON response
    try:
        response_json = json.loads(response_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None
    # Check if the response contains the expected keys
    if 'prediction' not in response_json:
        print("Error: 'prediction' key not found in the response.")
        return None
    cleaned_prediction = response_json['prediction'].replace('"', '')
    cleaned_prediction = cleaned_prediction.replace(',', ' and')
    return cleaned_prediction
