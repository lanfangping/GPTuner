from openai import OpenAI, APIError
import os
import re
import sys
import json
import time
import random
import datetime
import tiktoken
import transformers

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
        except APIError as e:
            print("Call API fail:", e)
            exit()
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
            chat_tokenizer_dir = "./src/knowledge_handler/deepseek_v3_tokenizer"
            enc =  transformers.AutoTokenizer.from_pretrained( 
                    chat_tokenizer_dir, trust_remote_code=True
                    )
        else:
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            # if self.model == 'gpt-4o-mini':
            #     try:
            #         enc = tiktoken.encoding_for_model(self.model)
            #     except KeyError:
            #         enc = tiktoken.get_encoding("cl100k_base")
            # else:
            #     enc = tiktoken.encoding_for_model(self.model)
            # enc = tiktoken.encoding_for_model(self.model)
            # enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(out_text+in_text))

    def calc_money(self, in_text, out_text):
        """money for gpt4"""
        if self.model == "gpt-4":
            return (self.calc_token(in_text) * 0.03 + self.calc_token(out_text) * 0.06) / 1000
        elif self.model == "gpt-3.5-turbo":
            return (self.calc_token(in_text) * 0.0015 + self.calc_token(out_text) * 0.002) / 1000
        elif self.model == "gpt-4-1106-preview" or self.model == "gpt-4-1106-vision-preview":
            return (self.calc_token(in_text) * 0.01 + self.calc_token(out_text) * 0.03) / 1000
        elif self.model == 'deepseek-chat':
            # input text: 0.14/1M, output text: 0.28/1M
            return (self.calc_token(in_text) * 0.14 + self.calc_token(out_text) * 0.28) / 1000000
        else:
            return 0 

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def _calculate_token_usage(self, token_usage):
        """
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "total_tokens": 0
        }
        """

        if "deepseek" in self.type:
            total_tokens = token_usage.total_tokens 
            completion_tokens = token_usage.completion_tokens
            prompt_tokens = token_usage.prompt_tokens
            prompt_cache_hit_tokens = token_usage.prompt_cache_hit_tokens
            prompt_cache_miss_tokens = token_usage.prompt_cache_miss_tokens
            current_time = datetime.now().strftime("%Y%m%d%H%M")
            if os.path.exists(os.path.join(self.usage_save_path, 'token_usage.txt')):
                with open(os.path.join(self.usage_save_path, 'token_usage.txt'), 'a') as f:
                    f.write(f"{current_time}, {total_tokens}, {completion_tokens}, {prompt_tokens}, {prompt_cache_hit_tokens}, {prompt_cache_miss_tokens}\n")
            else:
                with open(os.path.join(self.usage_save_path, 'token_usage.txt'), 'w') as f:
                    f.write(f"current_time, total_tokens, out_tokens, in_tokens, cache_hit_tokens, cache_miss_tokens\n")
                    f.write(f"{current_time}, {total_tokens}, {completion_tokens}, {prompt_tokens}, {prompt_cache_hit_tokens}, {prompt_cache_miss_tokens}\n")
        else:
            total_tokens = token_usage.total_tokens 
            completion_tokens = token_usage.completion_tokens
            prompt_tokens = token_usage.prompt_tokens
            current_time = datetime.now().strftime("%Y%m%d%H%M")
            if os.path.exists(os.path.join(self.usage_save_path, 'token_usage.txt')):
                with open(os.path.join(self.usage_save_path, 'token_usage.txt'), 'a') as f:
                    f.write(f"{current_time}, {total_tokens}, {completion_tokens}, {prompt_tokens}\n")
            else:
                with open(os.path.join(self.usage_save_path, 'token_usage.txt'), 'w') as f:
                    f.write(f"current_time, total_tokens, out_tokens, in_tokens\n")
                    f.write(f"{current_time}, {total_tokens}, {completion_tokens}, {prompt_tokens}\n")
    
