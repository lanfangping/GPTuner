from openai import OpenAI, APIError
import os
import re
import sys
import json
import time
import random
import tiktoken
import transformers
from datetime import datetime

class GPT:
    def __init__(self, api_base, api_key, model="gpt-4o-mini"):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.money = 0
        self.token = 0
        self.cur_token = 0
        self.cur_money = 0

    def get_GPT_response_json(self, prompt, json_format=True, n=3): # This function returns the GPT response, which can be specified to return json or string format
        if n <= 0:
            print("Call API failure.")
            exit()

        client = OpenAI(api_key=self.api_key, base_url = self.api_base)
        try:
            if json_format: # json
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You should output JSON."},
                        {'role':'user', 'content':prompt}],
                    model=self.model, 
                    response_format={"type": "json_object"}, 
                    temperature=0.5,
                )
                # print(response)
                ans = response.choices[0].message.content
                completion = json.loads(ans)  # Convert to json object
                
            else: # string
                response = client.chat.completions.create(
                    messages=[
                        {'role':'user', 'content':prompt}],
                    model=self.model, 
                    temperature=1,     
                )
                completion = response.choices[0].message.content

            self.calc_money(prompt, completion)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"Exception Type: {exc_type.__name__}")
            print(f"Exception Message: {str(e)}")
            print(f"Occurred at Line: {exc_tb.tb_lineno}")
            print("Sleeping...")
            time.sleep(random.randint(30, 40))
            print("retry.")
            return self.get_GPT_response_json(prompt, json_format, n-1)
        return completion
    
    def calc_token(self, in_text, out_text=""):
        if isinstance(in_text, dict):
            in_text = json.dumps(in_text)

        if isinstance(out_text, dict):
            out_text = json.dumps(out_text)

        if self.model == 'deepseek-chat':
            chat_tokenizer_dir = "./src/knowledge_handler/deepseek_v2_tokenizer"
            enc =  transformers.AutoTokenizer.from_pretrained( 
                    chat_tokenizer_dir, trust_remote_code=True
                    )
        else:
            enc = tiktoken.encoding_for_model(self.model)
            # enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(out_text+in_text))

    def calc_money(self, in_text, out_text):
        """money for gpt4"""
        save_path = 'optimization_results_tpch_sf1_t1_gpt4/token_usage.txt'
        if os.path.exists(save_path):
            record = open(save_path, 'a')
        else:
            record = open(save_path, 'w')
            record.write("current_time total_tokens in_tokens out_tokens\n")

        in_token = self.calc_token(in_text)
        out_token = self.calc_token(out_text)
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        record.write(f"{current_time} {in_token+out_token} {in_token} {out_token}\n")
        if self.model == "gpt-4":
            return (in_token * 0.03 + out_token * 0.06) / 1000
        elif self.model == "gpt-3.5-turbo":
            return (in_token * 0.0015 + out_token * 0.002) / 1000
        elif self.model == "gpt-4-1106-preview" or self.model == "gpt-4-1106-vision-preview":
            return (in_token * 0.01 + out_token * 0.03) / 1000
        elif self.model == 'deepseek-chat':
            # input text: 0.14/1M, output text: 0.28/1M
            return (in_token * 0.14 + out_token * 0.28) / 1000000
        else:
            return 0 

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def extract_json_from_text(self, text):
        # Regex to find the JSON block between ```json ... ```
        json_block = re.search(r"```json(.*?)```", text, re.DOTALL)
        if json_block:
            json_text = json_block.group(1).strip()  # Extract the JSON text inside ```json``` block
            try:
                # Parse the JSON text
                json_data = json.loads(json_text)
                return json_data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return None
        else:
            print("No JSON block found in the text.")
            return None
    
