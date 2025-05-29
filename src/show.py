import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

line_styles = [
    {'color': 'purple', 'linestyle': '-', 'marker': 'o'},
    {'color': 'teal', 'linestyle': '-', 'marker': 'v'},
    {'color': 'red', 'linestyle': '-', 'marker': '^'},
    {'color': 'navy', 'linestyle': '-', 'marker': 'v'},
    {'color': 'black', 'linestyle': '--', 'marker': 's'},
    {'color': 'green', 'linestyle': ':', 'marker': 'D'},
    {'color': 'blue', 'linestyle': '-.', 'marker': '*'},
    {'color': 'brown', 'linestyle': '--', 'marker': 'x'},
    {'color': 'orange', 'linestyle': '-', 'marker': 'o'},
    {'color': 'pink', 'linestyle': '-', 'marker': '^'},
]

def load_and_process(data, type):
    p = []
    for d in data:
        # print(d[4])
        if type == 'throughput':
            if d[4] >= 0:
                p.append(0)
            else:
                p.append(-d[4])
        elif type == 'latency':
            p.append(d[4])
        else:
            p.append(0)
    return p

def load_and_get_historical_best_data(folder_name, seed=100, type='throughput'):
    # print(folder_name)
    fine_path = f'./{folder_name}/postgres/fine/{seed}/runhistory.json' # tpcc sf20, t10
    coarse_path = f'./{folder_name}/postgres/coarse/{seed}/runhistory.json'
    performance = []
    historical_best_performance = []
    with open(coarse_path, 'r') as f:
        file_data = json.load(f)
        data = file_data['data']
        performance.extend(load_and_process(data, type))

    with open(fine_path, 'r') as f:
        file_data = json.load(f)
        data = file_data['data']
        performance.extend(load_and_process(data[30:], type))
    
    # print("performance", performance)
    if type == 'throughput':
        historical_best_performance = [max(performance[:i+1]) for i in range(len(performance))]
    else:
        historical_best_performance = [min(performance[:i+1]) for i in range(len(performance))]

    return historical_best_performance, performance

def load_and_get_data_with_deviation_from_project_data(project_name):
    project_data = json.load(open('experiments_results/tpcc/project_data.json', 'r'))
    data = []
    for performance_tuple in project_data[project_name]['performance_tuple']:
        data.append(performance_tuple[0])
    
    # Convert to numpy array for easier computation
    arr = np.array(data)

    # Calculate mean and standard deviation per index (column-wise)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)

    return means, stds

def show(data_lists:list, labels:list, output_file:str, num_plots:int=1, type:str='throughput'):
    # Re-plot with x-axis label changed to "Iteration"
    plt.figure(figsize=(12, 6))

    for i in range(num_plots):
        historical_best_performance, performance = data_lists[i]
        label = labels[i]
        style = line_styles[i%len(line_styles)]
        
        # Plot the line for past best throughputs
        plt.plot(range(len(historical_best_performance)), historical_best_performance, label=label, markevery=10, **style)

    # Updated x-axis label
    plt.xlabel('Iteration')
    plt.ylabel(type)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)

def show_with_std(data_lists:list, labels:list, output_file:str, num_plots:int=1, type:str='throughput'):
    # Re-plot with x-axis label changed to "Iteration"
    plt.figure(figsize=(12, 6))

    for i in range(num_plots):
        mean, std = data_lists[i]
        label = labels[i]
        style = line_styles[i%len(line_styles)]
        
        # Plot the line for past best throughputs
        plt.plot(range(len(mean)), mean, label=label, markevery=10, **style)
        # Plot the std deviation as a shaded area
        plt.fill_between(range(len(mean)), mean - std, mean + std, color=style['color'], alpha=0.2)
        # plt.errorbar(range(len(mean)), mean, yerr=std, fmt='-', capsize=3)
        plt.plot(range(len(mean)), mean + std, linestyle='--', color=style['color'], alpha=0.6)
        plt.plot(range(len(mean)), mean - std, linestyle='--', color=style['color'], alpha=0.6)

    # Updated x-axis label
    plt.xlabel('Iteration')
    plt.ylabel(type)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)

def maintain_data(folder="experiments_results/tpcc"):
    project_data = defaultdict(lambda: defaultdict(list))
    for folder_name in os.listdir(folder):
        if folder_name == 'figures' or folder_name == 'project_data.json':
            continue
        
        items = folder_name.split('_')
        if items[-1].isdigit():
            project_name = '_'.join(items[:-1])
            date_id = items[-1]
        else:
            project_name = folder_name
            date_id = ''

        data_tuple = None
        history_file = os.path.join(folder, folder_name)
        # print(history_file)
        try:
            data_tuple = load_and_get_historical_best_data(history_file)
        except FileNotFoundError as e:
            continue
        project_data[project_name]['performance_tuple'].append(data_tuple)
        project_data[project_name]['date_ids'].append(date_id)

    json.dump(project_data, open(os.path.join(folder, 'project_data.json'), 'w'))        


if __name__ == '__main__':
    maintain_data(folder="experiments_results/tpcc")
    # #==================overall================
    # num_plots = 7
    # files = [
    #     ("experiments_results/tpcc/deepseek-v3-overall_202504101721", 100),
    #     ("experiments_results/tpcc/deepseek-v3-overall_202504131541", 100),
    #     ("experiments_results/tpcc/gpt-4o-overall_202504102122", 100),
    #     ("experiments_results/tpcc/gpt-4o-overall_202504131536", 100), 
    #     ("experiments_results/tpcc/gpt4-4o-mini-overall_202504101933", 100),
    #     ("experiments_results/tpcc/gpt-4-previous-good-knowledge", 100),
    #     ("experiments_results/tpcc/gpt-3.5-turbo-overall_202504140003", 100),
    # ]
    # labels = [
    #     "Deepseek-v3-1", 
    #     "Deepseek-v3-2", 
    #     "GPT-4o-1", 
    #     "GPT-4o-2", 
    #     "GPT-4o-mini",
    #     "Previous Good(GPT-4)",
    #     "GPT-3.5-turbo"
    # ]
    # output_type = 'throughput'
    # output_file = "experiments_results/tpcc/figures/end_to_end_study/overall_compare_7.png"
    # data_lists = []
    # for i in range(num_plots):
    #     folder_name, seed = files[i]
    #     data = load_and_get_historical_best_data(folder_name=folder_name, seed=seed, type=output_type)
    #     data_lists.append(data)
    
    # show(data_lists=data_lists, labels=labels, output_file=output_file, num_plots=num_plots, type=output_type)

    #==================overall================
    # num_plots = 5
    # files = [
    #     ("experiments_results/tpcc/gpt-4-previous-good-knowledge", 100),
    #     ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt3.5turbo-st-gpt4-spv-gpt4_202505081057", 100),
    #     ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-st-gpt3.5turbo-spv-gpt4_202505081616", 100),
    #     ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-st-gpt4-spv-gpt3.5turbo_202505091059", 100), 
    #     ("experiments_results/tpcc/ks-gpt4-kr-gpt3.5turbo_202505011227", 100)
    # ]
    # labels = [
    #     "KS-GPT4, SV-GPT4, SR-GPT4, SPV-GPT4", 
    #     "KS-GPT4, SV-GPT3.5-turbo, SR-GPT4, SPV-GPT4",
    #     "KS-GPT4, SV-GPT4, SR-GPT3.5-turbo, SPV-GPT4",
    #     "KS-GPT4, SV-GPT4, SR-GPT4, SPV-GPT3.5-turbo",
    #     "KS-GPT4, SV-GPT3.5-turbo, SR-GPT3.5-turbo, SPV-GPT3.5-turbo"
    # ]

    num_plots = 3
    project_names = [
        "gpt-4-previous",
        "ks-gpt4-kr--sv-gpt4-sr-deepseekv3-spv-gpt4",
        "ks-gpt4-kr--sv-gpt4-sr-gpt4-narrow-deepseekv3-spv-gpt4",
        # ("experiments_results/tpcc/gpt-4-previous_mspv_202505270205", 100)
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-deepseekv3-spv-gpt4_202505080025", 100),
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-deepseekv3-spv-gpt4_202505201450", 100),
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-deepseekv3-spv-gpt4_202505210131", 100),
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-gpt4-narrow-deepseekv3-spv-gpt4_202505141738", 100),
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-gpt4-narrow-deepseekv3-spv-gpt4_202505151912", 100),
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-gpt4-narrow-deepseekv3-spv-gpt4_202505161241", 100),
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-gpt4-narrow-deepseekv3-spv-gpt4_202505171418", 100),
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-gpt4-narrow-deepseekv3-spv-gpt4_202505180328", 100),
        # ("experiments_results/tpcc/ks-gpt4-kr--sv-gpt4-sr-gpt4-narrow-deepseekv3-spv-gpt4_202505181352", 100)
    ]
    labels = [
        "KS-GPT4, SV-GPT4, SR-GPT4, SPV-GPT4", 
        "KS-GPT4, SV-GPT4, SR-Deepseekv3, SPV-GPT4", 
        "KS-GPT4, SV-GPT4, SR-Deepseekv3-narrow, SPV-GPT4",
        "DeepseekV3",
        "GPT3.5-turbo"
        # "KS-GPT4, SV-GPT4, SR-GPT4, SPV-manual"
        # "KS-GPT4, SV-GPT4, SR-Deepseekv3-1, SPV-GPT4",
        # "KS-GPT4, SV-GPT4, SR-Deepseekv3-2, SPV-GPT4",
        # "KS-GPT4, SV-GPT4, SR-Deepseekv3-3, SPV-GPT4",
        # "KS-GPT4, SV-GPT4, SR-GPT4-Deepseekv3-narrow-1, SPV-GPT4",
        # "KS-GPT4, SV-GPT4, SR-GPT4-Deepseekv3-narrow-2, SPV-GPT4",
        # "KS-GPT4, SV-GPT4, SR-GPT4-Deepseekv3-narrow-3, SPV-GPT4",
        # "KS-GPT4, SV-GPT4, SR-GPT4-Deepseekv3-narrow-4, SPV-GPT4",
        # "KS-GPT4, SV-GPT4, SR-GPT4-Deepseekv3-narrow-5, SPV-GPT4",
        # "KS-GPT4, SV-GPT4, SR-GPT4-Deepseekv3-narrow-6, SPV-GPT4",
    ]
    output_type = 'throughput'
    output_file = f"experiments_results/tpcc/figures/ablation_study/KR_study_std_{num_plots}.png"
    data_lists = []
    for i in range(num_plots):
        project_name = project_names[i]
        data = load_and_get_data_with_deviation_from_project_data(project_name=project_name)
        data_lists.append(data)
    
    show_with_std(data_lists=data_lists, labels=labels, output_file=output_file, num_plots=num_plots, type=output_type)

    

