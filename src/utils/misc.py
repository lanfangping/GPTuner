from ruamel.yaml import YAML
import re
import json

def over_write_args_from_file(args, yml_file):
    """
    overwrite arguments according to config file
    """
    if yml_file == '':
        return
    
    yaml = YAML(typ='rt')  # 'rt' is for round-trip parsing (preserves formatting)
    with open(yml_file, "r") as f:
        dic = yaml.load(f)
        for k in dic:
            setattr(args, k, dic[k])
    
def extract_json_knob_settings(text):
    # Regex to find the JSON block between ```json ... ```
    try:
        json_data = json.loads(text)
        return json_data
    except:
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

