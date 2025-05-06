import re
import os
import ast
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

def extract_suggested_values(folder_path):
    knob_suggested_values = {}
    for f in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, f)):
            knob = f.split('.')[0].strip()
            data = json.load(open(os.path.join(folder_path, f), 'r'))
            knob_suggested_values[knob] = data['suggested_values']
    return knob_suggested_values

def extract_knobs_from_different_sources(file_path):
    knobs_data = {
        'system':{'candidates':[], 'response':[]}, 
        'query': {'candidates':[], 'response':[]}, 
        'workload':{'candidates':[], 'response':[]}, 
        'interdependency': {}
    }
    with open(file_path, 'r') as f:
        lines = f.readlines()
        found_system_candidate_knobs = False
        found_workload_candidate_knobs = False
        found_query_candidate_knobs = False
        found_interdependent_candidate_knobs = False
        for line in lines:
            if "select_on_system_level" in line:
                if "th prompt:" in line:
                    found_system_candidate_knobs = True
                elif "th response" in line:
                    response_str = '{' + '{'.join(line.split('{')[1:])
                    response_dict = eval(response_str)
                    knobs_data['system']['response'].append(response_dict)
            elif "select_on_workload_level" in line: 
                if "th prompt:" in line:
                    found_workload_candidate_knobs = True
                elif "th response" in line:
                    response_str = '{' + '{'.join(line.split('{')[1:])
                    response_dict = eval(response_str)
                    knobs_data['workload']['response'].append(response_dict)
            elif "select_on_query_level" in line:
                if "th prompt:" in line:
                    found_query_candidate_knobs = True
                elif "th response" in line:
                    response_str = '{' + '{'.join(line.split('{')[1:])
                    response_dict = eval(response_str)
                    knobs_data['query']['response'].append(response_dict)
            elif "select_interdependent_all_knobs" in line:
                if "prompt:" in line:
                    found_interdependent_candidate_knobs = True
                elif "response" in line:
                    response_str = '{' + '{'.join(line.split('{')[1:])
                    response_dict = eval(response_str)
                    knobs_data['interdependency']['response'] = response_dict['knob_list']
            elif "Candidate knobs:" in line: 
                response_arr = eval('[' + '['.join(line.split('[')[1:]))
                if found_system_candidate_knobs:
                    knobs_data['system']['candidates'].append(response_arr)
                    found_system_candidate_knobs = False
                elif found_workload_candidate_knobs:
                    knobs_data['workload']['candidates'].append(response_arr)
                    found_workload_candidate_knobs = False
                elif found_query_candidate_knobs:
                    knobs_data['query']['candidates'].append(response_arr)
                    found_query_candidate_knobs = False
            elif "KNOB COLLECTION" in line and found_interdependent_candidate_knobs:
                response_arr = eval('[' + '['.join(line.split('[')[1:]))
                knobs_data['interdependency']['candidates'] = response_arr
                found_interdependent_candidate_knobs = False
    return knobs_data

if __name__ == '__main__':
    folder_path = "experiments_results/tpcc/ks-gpt4-kr-deepseekv3_202504301720"

    file_path = os.path.join(folder_path, 'ks-gpt4-kr-deepseekv3_log.txt')
    coarse_configuration, fine_configuration = extract_knob_range_from_file(file_path)
    print("=============Coarse search range===============")
    for key, info_dict in coarse_configuration.items():
        if 'special' in info_dict.keys():
            print(f"{key}|{info_dict['range']}|{info_dict['special']}")
        else:
            print(f"{key}|{info_dict['range']}|")

    print("\n\n=============Fine search range===============")
    for key, info_dict in fine_configuration.items():
        if 'special' in info_dict.keys():
            print(f"{key}|{info_dict['range']}|{info_dict['special']}")
        else:
            print(f"{key}|{info_dict['range']}|")
    
    print("\n\n============Suggested Values================")
    normal_path = os.path.join(folder_path, "knowledge_collection/postgres/structured_knowledge/normal") 
    knob_suggested_values = extract_suggested_values(normal_path)
    for knob, suggested_values in knob_suggested_values.items():
        print(f"{knob}|{suggested_values}")


    # print("\n\n============Selected Knobs================")
    # selection_file = os.path.join(folder_path, "knob_selection_log.txt") 
    # knobs_data = extract_knobs_from_different_sources(selection_file)
    # # print(knobs_data)
    # for level, data in knobs_data.items():
    #     print(f"\n============{level}================")
    #     if level != 'interdependency':
    #         for i, candidates in enumerate(data['candidates']):
    #             knob_importance = data['response'][i]
    #             missing_knobs = []
    #             print(f"============{i}th================")
    #             for candidate in candidates:
    #                 if candidate in knob_importance.keys():
    #                     print(f"{candidate}: {knob_importance[candidate]}")
    #                 else:
    #                     missing_knobs.append(candidate)
    #             print(f"============Missing================")
    #             print(missing_knobs)
    #             print(f"============Hallucinated================")
    #             candidates_set = set(candidates)
    #             response_set = set(knob_importance.keys())
    #             print(response_set.difference(candidates_set))

    #     else:
    #         print("Top 50 knobs:")
    #         print('\n'.join(data['candidates']))
    #         print(f"============================")
    #         print('\n'.join(data['response']))
        

            