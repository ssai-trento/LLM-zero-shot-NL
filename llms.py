import pandas as pd
import numpy as np
import json
import tqdm
import openai
import replicate
import os
import utils

os.environ["REPLICATE_API_TOKEN"] = "" #replicate token here
openai.api_key = "" #OpenAI token here




class LLM:
    def __init__(self, test_data, ground_data, prompt_type, model_name, trajectory_mode):
        self.dataset = test_data
        self.ground_data = ground_data
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.trajectory_mode = trajectory_mode
        self.prompts = []

        self.save_dir = "results/llm/" + self.model_name + "/" + self.prompt_type + "/"
        utils.create_dir(self.save_dir)

        self.hyperparams = {
            'temperature': 0.001,  # make the LLM basically deterministic
            'max_new_tokens': 100,
            'max_tokens': 1000,
        }

        if self.model_name not in ['gpt35turbo','gpt4', 'gpt4o', 'llama7b', 'llama13b', 'llama70b', 'llama7bchat', 'llama13bchat',
                                   'llama70bchat', 'mistral7b', 'llama3_8b', 'llama3_70b', 'llama3_8binstruct', 'llama3_70binstruct']:
            raise ValueError('Invalid model name! Please use one of the following: gpt35turbo,gpt4, gpt4o, llama7b, llama13b, llama70b, llama7bchat, '
                             'llama13bchat, llama70bchat, mistral7b, llama3_8b, llama3_70b, llama3_8binstruct, llama3_70binstruct')

        self.mapper = {
            'gpt35turbo': 'gpt-3.5-turbo',
            'gpt4': 'gpt-4-0613',
            'gpt4o': 'gpt-4-0613',
            'llama7b': 'meta/llama-2-7b:77dde5d6c56598691b9008f7d123a18d98f40e4b4978f8a72215ebfc2553ddd8',
            'llama13b': 'meta/llama-2-13b:078d7a002387bd96d93b0302a4c03b3f15824b63104034bfa943c63a8f208c38',
            'llama70b': 'meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00',
            'llama7bchat':'meta/llama-2-7b-chat:52551facc68363358effaacb0a52b5351843e2d3bf14f58aff5f0b82d756078c',
            'llama13bchat': 'meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d',
            'llama70bchat': 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
            'llama3_8b':  'meta/meta-llama-3-8b',
            'llama3_70b' :'meta/meta-llama-3-70b',
            'llama3_8binstruct': 'meta/meta-llama-3-8b-instruct',
            'llama3_70binstruct': 'meta/meta-llama-3-70b-instruct',
            'mistral7b': '"mistralai/mistral-7b-v0.1"'

        }

        self.generate_prompt()
        self.get_predictions()
        
    def generate_prompt(self):
        if self.trajectory_mode == 'trajectory_split':
            for k, v in self.dataset.items():
                # k is the user, v is a dictionary with traj_id : values
                for traj_id in v.keys():
                    self.prompts.append([k, traj_id, utils.prompt_generator(v[traj_id], self.prompt_type)])
        elif self.trajectory_mode == 'user_split':
            for k, v in self.dataset.items():
                self.prompts.append([k, utils.prompt_generator(v, self.prompt_type)])

    def get_predictions(self):
        if self.model_name == 'gpt35turbo' or self.model_name == 'gpt4turbo':
            for prompt in tqdm.tqdm(self.prompts):
                prompt_text = prompt[2]  #prompt text is the third element
                try:
                    response = openai.ChatCompletion.create(
                        model=self.mapper[self.model_name],
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant who predicts user next location."},
                            {"role": "user", "content": prompt_text}
                        ]
                    )

                    # Extract reply from the API response
                    full_text = response.choices[0].message["content"]
                    full_text = full_text.strip()
                    json_str = full_text[full_text.find('{'):full_text.rfind('}') + 1]

                    try:
                        output_json = json.loads(json_str)
                        prediction = output_json.get('prediction')
                        reason = output_json.get('reason')
                    except json.JSONDecodeError:
                        output_json = {"raw_response": full_text}
                        prediction = ""
                        reason = ""

                    true_value = self.ground_data[prompt[0]][prompt[1]] if len(prompt) == 3 else self.ground_data[prompt[0]]
                    predictions = {
                        'input': prompt_text,
                        'output': output_json,
                        'prediction': prediction,
                        'reason': reason,
                        'true': true_value  
                    }

                    # Construct the filename with model type and save to file
                    filename = f"{self.model_name}_{prompt[0]}_{prompt[1]}.json" if len(prompt) == 3 else f"{self.model_name}_{prompt[0]}.json"
                    file_path = os.path.join(self.save_dir, filename)
                    with open(file_path, 'w') as f:
                        json.dump(predictions, f, indent=4)

                except Exception as e:
                    print(f"An error occurred: {e}")

        else:
            print('Running LLM model: ' + self.model_name)
            for prompt in tqdm.tqdm(self.prompts):
                predictions = {}
                input = {
                    "max_new_tokens": 200,
                    "temperature": 0.01,
                    "prompt": prompt[-1]
                }

                # Generate response using the model
                iterator = replicate.run(self.mapper[self.model_name], input=input)
                full_text = ''.join([text for text in iterator])

                # Extract the JSON string from the full_text
                json_str = full_text[full_text.find('{'):full_text.rfind('}') + 1]

                # Attempt to load as JSON
                try:
                    output_json = json.loads(json_str)
                    prediction = output_json.get('prediction')
                    reason = output_json.get('reason')
                except json.JSONDecodeError:
                    # If not JSON, store the raw full_text string in a new dictionary
                    output_json = {
                        "raw_response": full_text
                    }
                    prediction = ""
                    reason = ""

                if len(prompt) == 3:
                    predictions = {
                        'input': prompt[-1],
                        'true': self.ground_data[prompt[0]][prompt[1]],
                        'output': output_json,
                        'prediction': prediction,
                        'reason': reason
                    }
                else:
                    predictions = {
                        'input': prompt[-1],
                        'true': self.ground_data[prompt[0]],
                        'output': output_json,
                        'prediction': prediction,
                        'reason': reason
                    }

                if len(prompt) == 3:
                    name = '_' + prompt[0] + '_' + prompt[1]
                else:
                    name = '_' + prompt[0]

                with open(self.save_dir + 'predictions_' + self.model_name + '_' + self.trajectory_mode + '_' + self.prompt_type + name + '.json', 'w') as f:
                    json.dump(predictions, f, indent=4)
