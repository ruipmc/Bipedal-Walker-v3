import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Diretório onde os logs estão armazenados
logs_dir = "logs"

# Algoritmos para comparar
algorithms = ["A2C_0", "ARS_0", "DDPG_0", "PPO_0", "RecurrentPPO_0", "SAC_0", "TQC_0", "TRPO_0"]

# Tempo total de treinamento em horas para cada algoritmo
training_times_hours = {
    "A2C_0": 1.472,
    "ARS_0": 0.395,  # 23.74 minutos convertidos para horas
    "DDPG_0": 6.493,
    "PPO_0": 1.529,
    "RecurrentPPO_0": 8.499,
    "SAC_0": 10.49,
    "TQC_0": 10.61,
    "TRPO_0": 2.797,
}

# Função para extrair o melhor reward
def extract_best_reward(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    rewards = []
    if "rollout/ep_rew_mean" in event_acc.Tags()["scalars"]:
        for event in event_acc.Scalars("rollout/ep_rew_mean"):
            rewards.append(event.value)
    
    # Encontrar o melhor reward
    best_reward = max(rewards) if rewards else None
    return best_reward

# Processar os logs de cada algoritmo
results = {}
for algo in algorithms:
    algo_dir = os.path.join(logs_dir, algo)
    if os.path.exists(algo_dir):
        best_reward = extract_best_reward(algo_dir)
        if best_reward and algo in training_times_hours:
            time_minutes = training_times_hours[algo] * 60  # Converter horas para minutos
            ratio = best_reward / time_minutes  # Calcular o ratio usando tempo em minutos
            results[algo] = {"best_ep_rew_mean": best_reward, "time_minutes": time_minutes, "ratio": ratio}

# Ordenar os algoritmos pelo ratio em ordem decrescente
sorted_results = sorted(results.items(), key=lambda x: x[1]["ratio"], reverse=True)

# Exibir os resultados
for algo, metrics in sorted_results:
    print(
        f"Algoritmo: {algo}, Time (min): {metrics['time_minutes']:.2f}, "
        f"Best_Ep_Rew_Mean: {metrics['best_ep_rew_mean']:.2f}, Ratio (Best_Ep_Rew_Mean/Time): {metrics['ratio']:.6f}"
    )
