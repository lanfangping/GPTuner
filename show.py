import json
import matplotlib.pyplot as plt

folder_name = "experiments_results/enhanced_fineD_modeAO_strategyR_202502072253"
file_path = f'./{folder_name}/postgres/fine/100/runhistory.json' # tpcc sf20, t10

performance = []
with open(file_path, 'r') as f:
    file_data = json.load(f)
    data = file_data['data']

    
    for d in data:
        # print(d[4])
        if d[4] >= 0:
            performance.append(0)
        else:
            performance.append(-d[4])

print("performance", performance)
past_best_performance = [max(performance[:i+1]) for i in range(len(performance))]
print("past_best_performance", past_best_performance)

# Re-plot with x-axis label changed to "Iteration"
plt.figure(figsize=(12, 6))

# Plot the line for past best throughputs
plt.plot(range(len(past_best_performance)), past_best_performance, color='blue', label='Past Best Throughput')

# Plot the scattered dots for actual throughput values in semi-transparency
plt.scatter(range(len(performance)), performance, color='red', marker='x', alpha=0.5, label='Actual Throughput')

# Updated x-axis label
plt.xlabel('Iteration')
plt.ylabel('Throughput')
plt.title('Throughput with Past Best and Actual Values')
plt.legend()
plt.grid(True)
plt.savefig(f'./{folder_name}/figure_tpcc_fine_sf20_t10.png')