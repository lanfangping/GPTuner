import re
import json
from collections import defaultdict

def extract_knob_range_from_file(file_path):
    coarse_configuration = defaultdict(dict)
    fine_configuration = defaultdict(dict)
    coarse_start = False
    fine_start = False
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Coarse Configuration Space" in line:
                coarse_start = True
                
            elif "Fine Configuration Space" in line:
                fine_start = True
            elif "Conditions:" in line:
                coarse_start = False
                fine_start = False

            elif coarse_start and "Hyperparameters:" not in line:
                knob, knob_range = extract_range_from_text(line)
                if knob_range is None:
                    continue
                elif knob.startswith('special_'):
                    knob = knob[8:]
                    coarse_configuration[knob]['special'] = knob_range
                else:
                    coarse_configuration[knob]['range'] = knob_range # knob_range may be a value
            elif fine_start and "Hyperparameters:" not in line:
                knob, knob_range = extract_range_from_text(line)
                if knob_range is None:
                    continue
                elif knob.startswith('special_'):
                    knob = knob[8:]
                    fine_configuration[knob]['special'] = knob_range
                else:
                    fine_configuration[knob]['range'] = knob_range # knob_range may be a value

    return coarse_configuration, fine_configuration
                
def extract_range_from_text(line):
    knob = line.split(',')[0].strip()
    if knob.startswith('control_'):
        return knob, None

    if "Type: UniformInteger" in line:
        # print("Integer Line:", line)
        match = re.search(r'Range:\s*\[(-?\d+),\s*(\d+)\]', line)
        if match:
            range_values = list(map(int, match.groups()))
            # print(f"{knob}: Range {range_values}")
            return knob, range_values
    elif "Type: UniformFloat" in line:
        # print("Float Line:", line)
        match = re.search(r'Range:\s*\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]', line)
        if match:
            range_values = list(map(float, match.groups()))
            # print(f"{knob}: Range {range_values}")
            return knob, range_values
        else:
            print(f"Line: {line} may contain parameter-range pair but the pattern cannot capture.")
    elif "Type: Categorical" in line:
        match = re.search(r'Choices:\s*\{(.*?)\}', line)
        if match:
            choices = [x.strip() for x in match.group(1).split(',')]
            # print(f"{knob}: Choices {choices}")
            return knob, choices
        else:
            print(f"Line: {line} may contain parameter-range pair but the pattern cannot capture.")
    elif "Type: Constant" in line:
        match = re.search(r'Value:\s*([-+]?[0-9]*\.?[0-9]+)', line)
        if match:
            value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
            # print(f"{knob}: Constant {value}")
            return knob, value
        else:
            print(f"Line: {line} may contain parameter-range pair but the pattern cannot capture.")
    return knob, None

if __name__ == '__main__':
    file_path = 'experiments_results/tpcc/deepseek-v3-overall_202504101721/deepseek-v3-overall_log.txt'
    coarse_configuration, fine_configuration = extract_knob_range_from_file(file_path)
    # print("coarse_configuration", coarse_configuration)
    # print("fine_configuration", fine_configuration)

    for key, info_dict in coarse_configuration.items():
        if 'special' in info_dict.keys():
            print(f"{key}|{info_dict['range']}|{info_dict['special']}")
        else:
            print(f"{key}|{info_dict['range']}|")

    print("\n\n============================")
    for key, info_dict in fine_configuration.items():
        if 'special' in info_dict.keys():
            print(f"{key}|{info_dict['range']}|{info_dict['special']}")
        else:
            print(f"{key}|{info_dict['range']}|")

    # line = "autovacuum_vacuum_cost_limit, Type: UniformInteger, Range: [-1, 10000], Default: -1"
    # match = re.search(r'Range:\s*\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]', line)
    # print(match)
    # range_values = list(map(float, match.groups()))
    # print(range_values)
