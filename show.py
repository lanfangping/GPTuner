import json
import matplotlib.pyplot as plt

folder_name = "optimization_results/run_202503271344"
performance_type = 'throughput'

def load_and_process(data):
    p = []
    for d in data:
        # print(d[4])
        if performance_type == 'throughput':
            if d[4] >= 0:
                p.append(0)
            else:
                p.append(-d[4])
        elif performance_type == 'latency':
            p.append(d[4])
        else:
            p.append(0)
    return p

def integration():
    seeds=[80, 90, 100]
    colors = ['blue', 'yellow', 'red']

    # Re-plot with x-axis label changed to "Iteration"
    plt.figure(figsize=(12, 6))

    for i, seed in enumerate(seeds):
        file_path = f'./{folder_name}/postgres/fine/{seed}/runhistory.json' # tpcc sf20, t10
        coarse_path = f'./{folder_name}/postgres/coarse/{seed}/runhistory.json'

        performance = []
        with open(coarse_path, 'r') as f:
            file_data = json.load(f)
            data = file_data['data']
            performance.extend(load_and_process(data))

        with open(file_path, 'r') as f:
            file_data = json.load(f)
            data = file_data['data']
            performance.extend(load_and_process(data[30:]))

        # print("performance", performance)
        if performance_type == 'throughput':
            past_best_performance = [max(performance[:i+1]) for i in range(len(performance))]
        else:
            past_best_performance = [min(performance[:i+1]) for i in range(len(performance))]
        # print("past_best_performance", past_best_performance)

        # Plot the line for past best throughputs
        plt.plot(range(len(past_best_performance)), past_best_performance, color=colors[i], label=f'seed={seed}')

        # Plot the scattered dots for actual throughput values in semi-transparency
        # plt.scatter(range(len(performance)), performance, color='red', marker='x', alpha=0.5, label='Actual Throughput')

    # Updated x-axis label
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    plt.title(f'Throughput with Past Best {performance_type}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{folder_name}/figure_{performance_type}_compare.png')

def show(seed=100):
    # Re-plot with x-axis label changed to "Iteration"
    plt.figure(figsize=(12, 6))

    file_path = f'./{folder_name}/postgres/fine/{seed}/runhistory.json' # tpcc sf20, t10
    coarse_path = f'./{folder_name}/postgres/coarse/{seed}/runhistory.json'

    performance = []
    with open(coarse_path, 'r') as f:
        file_data = json.load(f)
        data = file_data['data']
        performance.extend(load_and_process(data))

    with open(file_path, 'r') as f:
        file_data = json.load(f)
        data = file_data['data']
        performance.extend(load_and_process(data[30:]))

    # print("performance", performance)
    if performance_type == 'throughput':
        past_best_performance = [max(performance[:i+1]) for i in range(len(performance))]
    else:
        past_best_performance = [min(performance[:i+1]) for i in range(len(performance))]
    # print("past_best_performance", past_best_performance)

    # Plot the line for past best throughputs
    plt.plot(range(len(past_best_performance)), past_best_performance, color='blue', label=f'seed={seed}')

    # Plot the scattered dots for actual throughput values in semi-transparency
    # plt.scatter(range(len(performance)), performance, color='red', marker='x', alpha=0.5, label='Actual Throughput')

    # Updated x-axis label
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    plt.title(f'Throughput with Past Best {performance_type}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{folder_name}/figure_{performance_type}_{seed}.png')

if __name__ == '__main__':
    # integration()
    show(100)