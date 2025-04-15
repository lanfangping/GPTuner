import json
import matplotlib.pyplot as plt

line_styles = [
    {'color': 'purple', 'linestyle': '-', 'marker': 'o'},
    {'color': 'orange', 'linestyle': '-', 'marker': 'o'},
    {'color': 'darkorange', 'linestyle': '-', 'marker': '^'},
    {'color': 'teal', 'linestyle': '-', 'marker': 'v'},
    {'color': 'red', 'linestyle': '-', 'marker': '^'},
    {'color': 'navy', 'linestyle': '-', 'marker': 'v'},
    {'color': 'black', 'linestyle': '--', 'marker': 's'},
    {'color': 'green', 'linestyle': ':', 'marker': 'D'},
    {'color': 'blue', 'linestyle': '-.', 'marker': '*'},
    {'color': 'brown', 'linestyle': '--', 'marker': 'x'}
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

def show(data_lists:list, labels:list, output_file:str, num_plots:int=1, type:str='throughput'):
    # Re-plot with x-axis label changed to "Iteration"
    plt.figure(figsize=(12, 6))

    for i in range(num_plots):
        historical_best_performance, performance = data_lists[i]
        label = labels[i]
        style = line_styles[i%len(line_styles)]
        
        # Plot the line for past best throughputs
        plt.plot(range(len(historical_best_performance)), historical_best_performance, label=label, markevery=10, **style)

        # # Plot the scattered dots for every 10 iterations
        # dots = []
        # dots_index = []
        # for i in range(len(historical_best_performance)):
        #     if i % 10 == 0:
        #         dots_index.append(i)
        #         dots.append(historical_best_performance[i])

        # if len(historical_best_performance) % 10 != 0:
        #     dots.append(historical_best_performance[-1]) # add the last dot when the number of data is not 10x
        #     dots_index.append(len(historical_best_performance)-1)
        # plt.scatter(dots_index, dots, color=style['color'], marker=style['marker'])

    # Updated x-axis label
    plt.xlabel('Iteration')
    plt.ylabel(type)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)

if __name__ == '__main__':
    
    num_plots = 6
    files = [
        ("experiments_results/deepseek-v3-overall_202504101721", 100),
        ("experiments_results/gpt-4o-overall_202504102122", 100),
        ("experiments_results/gpt-4o-overall_202504131536", 100), 
        ("experiments_results/gpt4-4o-mini-overall_202504101933", 100),
        ("experiments_results/gpt-4-previous-good-knowledge", 100),
        ("experiments_results/gpt-3.5-turbo-overall_202504140003", 100),
    ]
    labels = [
        "Deepseek-v3", 
        "GPT-4o-1", 
        "GPT-4o-2", 
        "GPT-4o-mini",
        "Previous Good(GPT-4)",
        "GPT-3.5-turbo"
    ]
    output_type = 'throughput'
    output_file = "experiments_results/figures/end_to_end_study/overall_compare_6.png"
    data_lists = []
    for i in range(num_plots):
        folder_name, seed = files[i]
        data = load_and_get_historical_best_data(folder_name=folder_name, seed=seed, type=output_type)
        data_lists.append(data)
    
    show(data_lists=data_lists, labels=labels, output_file=output_file, num_plots=num_plots, type=output_type)

    