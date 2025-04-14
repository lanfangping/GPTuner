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
import torch
from huggingface_hub import login
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer


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
    
class LLM:
    def __init__(self, access_token, model='llama3-8b'):
        self.money = 0
        self.token = 0
        self.cur_token = 0
        self.cur_money = 0
        self.model_ids = {
            'llama3-8b': "meta-llama/Meta-Llama-3-8B-Instruct"
        }
        self._load_LLM_model(access_token=access_token, model=model)


    def _load_LLM_model(self, access_token, model):
        login(access_token)
        model_id = self.model_ids[model.lower()]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        # Set pad_token to eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
    
    def get_GPT_response_json(self, prompt, json_format=True, n=3):
        if n <= 0:
            print("Fail to get response.")
            exit()

        if json_format:
            messages = [
                {"role": "system", "content": "You should output JSON."},
                {"role": "user", "content": f"{prompt}"},
            ]
            temperature = 0.5
        else:
            messages = [
                {"role": "user", "content": f"{prompt}"},
            ]
            temperature = 1
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(self.llm.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        try:
            with torch.no_grad():
                outputs = self.llm.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=terminators,
                    temperature=temperature,
                    use_cache=False
                )
            response_tokens = outputs[0][input_ids.shape[-1]:]
            print(f"input tokens: {len(input_ids[0])}")
            print(f"output tokens: {len(response_tokens)}")
            self.cur_token = len(response_tokens) + len(input_ids[0])
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            print("response:", response)
            if json_format:
                json_response = self.extract_json_from_response(response)
                if json_response is None:
                    return self.get_GPT_response_json(prompt=prompt, json_format=json_format, n=n-1)
                else:
                    return json_response
            return response
        except Exception as e:
            print(f"LLM error: {e}")
            print(f"The Input token length: {len(input_ids[0])}")
            exit()
        finally:
            torch.cuda.empty_cache()

    def calc_token(self):
        return self.cur_token

    def calc_money(self):
        return 0

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def extract_json_from_response(self, text):
        # Regex to find the JSON block between ```json ... ```
        try:
            json_data = json.loads(text) # test whether the response is pure json
            return json_data
        except: # if the response mix up the text and the json, extract the json
            json_block = re.search(r"```(.*?)```", text, re.DOTALL)
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
                # If no backtick block, attempt to find balanced curly braces
                start = text.find('{')
                if start == -1:
                    return None

                brace_count = 0
                for i in range(start, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start:i+1]
                            try:
                                # Verify it's valid JSON
                                json_data = json.loads(json_str)
                                return json_data
                            except json.JSONDecodeError:
                                return None
                print("No JSON block found in the text.")
                return None
    
    

if __name__ == '__main__':
    model = LLM(access_token="", model='llama3-8b')
    prompt = """
 Suppose you are an experienced DBA, and you are required to tune a knob of postgres.

            TASK DESCRIPTION:
            Given the knob name along with its suggestion and hardware information, your job is to offer three values that may lead to the best performance of the system and meet the hardware resource constraints. The three values you need to provide are 'suggested_values', 'min_values', and 'max_values'. If you can identify one or more exact discrete suggested values, treat them as 'suggested_values'. If the suggested values fall within a continuous interval, provide the 'min_value' and 'max_value' for that interval.

            Note that the result you provide should be derived or inferred from the information provided. The result values should be numerical, and if a unit is needed, you can only choose from [KB, MB, GB, ms, s, min]; other units are not permitted.

            The question you need to solve will be given in the HTML tag <question>, the suggested steps to follow to finish the job are in <step>, and some examples will be given in the <example> tag.

            <step>
            Step 1: Check if the suggestion provides values for the knob; if so, identify the relevant sentences and move to Step 2. If not, move to Step 2. Note that there may be several sentences you should try to find them all.
            Step 2: Check if the suggestion recommends some values related to hardware information. If so, proceed to Step 3; if not, proceed to Step 4.
            Step 3: Read the hardware information to figure out the hardware-relevent value(s); some easy computation may be required.
            Step 4: Check whether the suggestion offers a specific recommended value or a recommended range for good performance or both of them. Note that sometimes the default value or the permitted value range of the knob is given, but these are not the recommended values for optimal DBMS performance, so ignore these values.
            Step 5: If discrete suggested values are given, list them under 'suggested_values'.
            Step 6: If a suggested range is given, set the upper and lower bounds of the range as the 'max_value' and 'min_value', respectively.
            Step 7: Return the result in JSON format.
            </step>

            <EXAMPLES>

            <example>
KNOB: shared_buffers
SUGGESTION:
The 'shared_buffers' parameter determines the amount of memory used by the database server for shared memory buffers. It's advisable to set 'shared_buffers' to 25% of the system's memory for systems with 1GB or more RAM, but not exceeding 40% to allow PostgreSQL to rely on the operating system cache. For systems with less than 1GB of RAM, a smaller percentage is appropriate to leave adequate space for the operating system. However, any larger settings would necessitate an increase in 'max_wal_size'. Remember that each Postgres instance will reserve its own memory allocations and this variable is directly related to OS kernel parameters `shmmax` and `shmall`. If your system has hundreds of GBs, consider setting up huge pages. It's also recommended to conduct your own benchmarks varying this parameter and adjust accordingly, especially for non-dedicated servers.

ANS:
Step 1: The suggestion provides values for the knob 'shared_buffers'. It suggests setting 'shared_buffers' to 25% of the system's memory for systems with 1GB or more RAM, but not exceeding 40% to allow PostgreSQL to rely on the operating system cache. For systems with less than 1GB of RAM, a smaller percentage is appropriate. Proceed to Step 2.
Step 2: The suggestion recommends values related to hardware information, specifically the system's memory. Proceed to Step 3.
Step 3: The hardware information indicates that the machine has a RAM of 110 GB. Therefore, the 'shared_buffers' parameter should be set to 25% of 110 GB, which is 27.5 GB, but not exceeding 40% of 110 GB, which is 44 GB.
Step 4: The suggestion offers a recommended discrete value as "25% of the system's memory for systems with 1GB or more RAM," and a range for good performance, which is not exceeding 40% of the system's memory.
Step 5: 25% of the system's memory is proposed as a discrete suggested value. Therefore, "27.5 GB" is put into the "suggested_values".
Step 6: The suggested range is not exceeding 40% of the system's memory. Therefore, the the 'max_value' is "44 GB".
Step 7: Return the result in JSON format. The result is:
{
"suggested_values": ["27.5 GB"],
"min_value": null,
"max_value": "44 GB"
}
<\example>
<example>
KNOB: vacuum_cost_delay
SUGGESTIONS:
- The 'vacuum_cost_delay' knob can be set to a value between 0 and 100 milliseconds, where a lower value will make the vacuum process faster but consume more resources, while a higher value will slow down the vacuum process but consume fewer resources.
- The 'vacuum_cost_delay' knob refers to the pause duration when the cost limit is exceeded during a process, measured in milliseconds by default, with a default value of zero disabling the cost-based vacuum delay feature, and positive values enabling it; optimal values are typically less than 1 millisecond, but larger delays may not be accurately measured on older platforms.

ANS:
Step 1: The suggestion offers a recommended value as "optimal values are typically less than 1 millisecond." Proceed to Step 2.
Step 2: The knob is not related to hardware information; proceed to Step 4.
Step 4: The suggestion offers a recommended range rather than specific discrete values. Proceed to Step 5.
Step 5: No discrete suggested values are given.
Step 6: The range is "less than 1 millisecond." Only the upper bound of the recommended range is given. Since the unit is millisecond, which corresponds to "ms" in the permitted units, set "1 ms" as "max_value". Although we can infer that the lower bound permitted for the knob is 0, the recommended lower bound is not given, so leave it alone.
Step 7: Return the result in JSON format, i.e.:
{{
"suggested_values": [],
"min_value": null,
"max_value": "1 ms"
}}
<\example>
<example>
KNOB: max_wal_size
SUGGESTIONS:
Based on the workload and available disk space, it is suggested to adjust the 'max_wal_size' value. Typically, this is set to about 1GB, but for heavier workloads, it can be increased to 4GB or more. However, unless there are disk space constraints, it is recommended to raise this value to ensure automatic checkpoints are typically caused by timeout and not by disk space. It's important to note that the 'max_wal_size' parameter controls the maximum size the WAL can grow during automatic checkpoints, with a soft limit default of 1 GB, which can be exceeded under certain conditions like heavy load or high wal_keep_size setting. Be aware that increasing this parameter may extend the time required for crash recovery.

ANS:
Step 1: The suggestion offers a recommended value as "Typically, this is set to about 1GB, but for heavier workloads, it can be increased to 4GB or more." Proceed to Step 2.
Step 2: The knob is not related to hardware information; proceed to Step 4.
Step 4: The suggestion recommends two specific discrete values as "Typically, this is set to about 1GB" and "it can be increased to 4GB or more." Note that "4 GB" does not provide a bound of a range, so it should be a suggested_value. No suggested range is given.
Step 5: No suggested range is given.
Step 6: The discrete suggested values are "1GB" and "4GB".
Step 7: Return the result in JSON format, i.e.:
{
    "suggested_values": ["1 GB", "4 GB"],
    "min_value": null,
    "max_value": null
}
<\example>

            </EXAMPLES>

            <question>
            KNOB: force_parallel_mode
            SUGGESTION: {"manual_suggestion": "Here is a summarized sentence associated with concrete numbers: We should consider using a combination of 30% manual testing and 70% automated testing to ensure optimal efficiency."}
            HARDWARE INFORMATION: The machine running the dbms has a RAM of 503 GB, a CPU of 24 cores, and a 1864 GB HDD drive.
            JSON RESULT TEMPLATE:
            {
                "suggested_values": [], // these should be exact values with a unit if needed (allowable units: KB, MB, GB, ms, s, min)
                "min_value": null,      // change it if there is a hint about the minimum value in SUGGESTIONS
                "max_value": null      // change it if there is a hint about the maximum value in SUGGESTIONS, it should be larger than min_value

            }
            </question>

            Let us think step by step and finally provide me with the result in JSON format. If no related information is provided in suggestions, just keep the result values at their default.
"""
    response = model.get_GPT_response_json(prompt=prompt, json_format=True)
    print(response)
    


