import json
from collections import defaultdict

def get_config(tag):
    # memory_file_path = f'../reflexion_memory/memory_{tag}.json'
    # output_file_path = f'../historical_best_config/historical_best_{tag}.json'
    memory_file_path = f'../results/runs/run_{tag}/memory.json'
    output_file_path = f'../historical_best_config/historical_best_{tag}.json'

    memory_file = json.load(open(memory_file_path, 'r'))
    recent_settings_history = memory_file['recent_settings_history']

    historical_best_configs = {}
    performance = 0
    iteration = 0
    for idx, data in recent_settings_history.items():
        if performance < data['performance']:
            performance = data['performance']
            historical_best_configs[iteration] = {
                'performance': performance,
                'config': data['settings']
            }
            iteration += 1

    json.dump(historical_best_configs, open(output_file_path, 'w'))

def get_ordered_configs(tag, top_n=None):
    ''
    memory_file_path = f'../results/runs/run_{tag}/memory.json'
    if top_n is None:
        output_file_path = f'../historical_ordered_config/historical_ordered_{tag}.json'
    else:
        output_file_path = f'../historical_ordered_config/historical_ordered_top{top_n}_{tag}.json'

    memory_file = json.load(open(memory_file_path, 'r'))
    recent_settings_history = memory_file['recent_settings_history']

    historical_configs = {}
    performance = 0
    iteration = 0
    for idx, data in recent_settings_history.items():
        performance = data['performance']
        historical_configs[iteration] = {
            'performance': performance,
            'config': data['settings']
        }
        iteration += 1

    # Sort by performance
    if top_n is None:
        sorted_configs = dict(sorted(historical_configs.items(), key=lambda x: x[1]['performance'], reverse=True))
    else:
        sorted_configs = dict(sorted(historical_configs.items(), key=lambda x: x[1]['performance'], reverse=True)[:top_n])
    
    json.dump(sorted_configs, open(output_file_path, 'w'))
    

def get_adjustment_from_historical_best_configs(tag):
    file_path = f'../historical_best_config/historical_best_{tag}.json'
    historical_best_configs = json.load(open(file_path, 'r'))
    output_data = []
    last_settings = {}
    for _id, data in historical_best_configs.items():
        settings = data['config']
        if settings == 'default settings':
            continue

        output_data.append(f"\n\n{_id} config:===============================\n")
        output_data.append(f"\n\nPerformance: {data['performance']}\n")
        
        knobs_set = set(settings.keys())
        last_knobs_set = set(last_settings.keys())
        deleted_set = last_knobs_set.difference(knobs_set)
        added_set = knobs_set.difference(last_knobs_set)
        output_data.append("\n***>>>>>>>>Added knobs compared to the last setting")
        for knob in added_set:
            output_data.append(f"{knob} = {settings[knob]}")
        output_data.append("\n***>>>>>>>>Deleted knobs compared to the last setting")
        for knob in deleted_set:
            output_data.append(f"{knob}")

        overlapped_knobs = last_knobs_set.intersection(knobs_set)
        output_data.append("\n***>>>>>>>>knob values adjustment compared to the last setting")
        for knob in overlapped_knobs:
            if f"{last_settings[knob]}" != f"{settings[knob]}":
                output_data.append(f"{knob} = {last_settings[knob]} -> {settings[knob]}")
        output_data.append(f"\n\nPerformance: {data['performance']}\n")

        last_settings = settings

    with open(f'../historical_best_config/adjustment_{tag}.txt', 'w') as f:
        f.write('\n'.join(output_data))

def get_knob_frenquency():
    configs_dict = json.load(open("enhanced_configurations/tpcc_sf20_t10/historical_best_config/historical_ordered_tpcc_sf20_t10_newflow_newimp_SR10_M8_Binary_IS1_TP8_IN0__202503152018.json", 'r'))
    config_history = []
    knob_count_dict = defaultdict(int)
    for _, config_info in configs_dict.items():
        if type(config_info['config']) is str:
            break
        if config_info['config'] in config_history:
            continue
        else:
            for knob, value in config_info['config'].items():
                knob_count_dict[knob] += 1
    
    print(f"there are {len(configs_dict)} duplicate configurations")
    for knob, freq in knob_count_dict.items():
        print(f"{knob}: {freq}")

if __name__ == '__main__':
    # tag = 'tpcc_sf20_t10_newflow_newimp_SR10_M8_Binary_IS1_TP8_IN0__202503152018'
    # # get_ordered_configs(tag)
    # get_config(tag)
    # get_adjustment_from_historical_best_configs(tag)
    get_knob_frenquency()