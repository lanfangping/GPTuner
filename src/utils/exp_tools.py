import os
import re
import json

def _transfer_unit(value):
        value = str(value)
        value = value.replace(" ", "")
        value = value.replace(",", "")
        if value.isalpha():
            value = "1" + value
        # pattern = r'(\d+\.\d+|\d+)([a-zA-Z]+)'
        pattern = r'(\d+(?:\.\d+)?)[\s]*([a-zA-Z]+)'
        match = re.match(pattern, value)
        if not match:
            return float(value)
        number, unit = match.group(1), match.group(2)
        unit_to_size = {
            'kB': 1e3,
            'KB': 1e3,
            'MB': 1e6,
            'GB': 1e9,
            'TB': 1e12,
            'K': 1e3,
            'M': 1e6,
            'G': 1e9,
            'B': 1,
            'ms': 1,
            's': 1000,
            'min': 60000,
            'h': 60 * 60000,
            'hour': 60 * 60000,
            'day': 24 * 60 * 60000,
            'million': 1e6
        }
        if unit in unit_to_size.keys():
            return float(number) * unit_to_size[unit]
        else:
            print(f"unknown unit '{unit}', return 1")
            return 1
    
def _type_transfer(knob_type, value):
    value = str(value)
    value = value.replace(",", "")
    if knob_type == "integer":
        return int(round(float(value)))
    if knob_type == "real":
        return float(value)

def replace_range_for_knobs(target_knobs, source_folder, target_folder, strategy='narrow'):
    """
    strategy = `narrow`
        replace range in source_folder with narrow range in target_folder
        for example, work_mem = [1, 100] in `source folder`, but work_mem=[1,64] in `target_folder`,
        then work_mem=[1, 64] and save it to `save_folder`.
    """
    system_info = json.load(open(os.path.join(target_folder, "knowledge_collection/postgres/knob_info/system_view.json")))

    knob_ranges = {}
    for knob in target_knobs:
        source_data = json.load(open(os.path.join(source_folder, 'knowledge_collection/postgres/structured_knowledge/normal', f'{knob}.json'), 'r'))
        target_data = json.load(open(os.path.join(target_folder, 'knowledge_collection/postgres/structured_knowledge/normal', f'{knob}.json'), 'r'))

        source_min = source_data["min_value"]
        target_min = target_data['min_value']

        source_max = source_data["max_value"]
        target_max = target_data['max_value']

        min_value = None
        max_value = None
        vartype = system_info[knob]['vartype']
        unit = system_info[knob]['unit']

        # print("before tranferring:", source_min, target_min, source_max, target_max)
        # for value in [source_min, target_min, source_max, target_max]: 
        def transfer(unit, value):
            if value is not None:
                if unit:
                    unit = _transfer_unit(unit)
                    value = _transfer_unit(value) / unit
                else:
                    value = _transfer_unit(value)
                    value = _type_transfer(vartype, value)
            return value
        
        source_min = transfer(unit, source_min)
        target_min = transfer(unit, target_min)
        source_max = transfer(unit, source_max)
        target_max = transfer(unit, target_max)

        # print("after transferring:", source_min, target_min, source_max, target_max)

        if strategy == 'narrow':
            if source_min is not None and target_min is not None:
                if source_min < target_min:
                    min_value = source_data["min_value"]
                else:
                    min_value = target_data["min_value"]
            elif source_min is not None:
                min_value = source_data["min_value"]
            elif target_min is not None:
                min_value = target_data["min_value"]
            
            if source_max is not None and target_max is not None:
                if source_max < target_max:
                    max_value = source_data["max_value"]
                else:
                    max_value = target_data["max_value"]
            elif source_max is not None:
                max_value = source_data["max_value"]
            elif target_max is not None:
                max_value = target_data["max_value"]
            
        
        knob_ranges[knob] = {
            'min_value': min_value,
            'max_value': max_value
        }
    
    return knob_ranges

def replace_special_values(target_knobs, source_folder, source_base_folder, target_folder):
    for knob in target_knobs:
        file_name = f"{knob}.json"
        print(file_name)
        if file_name in os.listdir(source_base_folder):
            with open(os.path.join(source_base_folder, file_name), 'r') as json_file:
                special_skill = json.load(json_file)
                special_knob = special_skill["special_knob"]
                if type(special_knob) == str and special_knob.lower() == 'true' or special_knob is True:
                    # source_special = True
                    # source_special_value = special_skill["special_value"]
                    replace_value_json = json.load(open(os.path.join(source_folder, file_name), 'r'))
                    json.dump(replace_value_json, open(os.path.join(target_folder, file_name), 'w'))
                else:
                    json.dump(special_skill, open(os.path.join(target_folder, file_name), 'w'))



def generate_special_value_json_file(output_folder):
    manual_set_special_values_json = json.load(open("knowledge_collection/postgres/special_values.json", "r"))

    for knob, special_value in manual_set_special_values_json.items():
        if special_value is None:
            json.dump({"special_knob": True, "special_value": special_value}, open(os.path.join(output_folder, f"{knob}.json"), 'w'))
        else:
            json.dump({"special_knob": False, "special_value": special_value}, open(os.path.join(output_folder, f"{knob}.json"), 'w'))


if __name__ == '__main__':
    generate_special_value_json_file("knowledge_collection/postgres/manual_collected_special")