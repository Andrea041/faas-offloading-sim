import json
import matplotlib.pyplot as plt
import numpy as np

def loss_plot(data):
    if "loss" in data:
        loss_values = data["loss"]
    else:
        print("Errore: La chiave 'loss' non è presente nel file JSON.")
        exit()

    iterations = list(range(1, len(loss_values) + 1))

    # Calcolo media cumulativa
    cumulative_avg_loss = np.cumsum(loss_values) / np.arange(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 6))
    # Curva originale della loss
    plt.plot(iterations, loss_values, label="Loss", alpha=0.5)
    # Curva della media cumulativa
    plt.plot(iterations, cumulative_avg_loss, label="Loss Media Cumulativa", linewidth=1, color='red')

    plt.title("Loss con Media Cumulativa")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def reward_plot(data):
    # Estrae i reward
    rewards = data["reward"]

    if not rewards:
        raise ValueError("Nessun campo 'reward' trovato nel file JSON.")

    episodes = range(1, len(rewards) + 1)

    cumulative_avg = np.cumsum(rewards) / episodes

    # Grafico
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, cumulative_avg, label='Media Cumulativa', color='blue', linewidth=1)

    plt.xlabel('Episodio')
    plt.ylabel('Reward')
    plt.title('Andamento Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def actions_plot(data):
    # Prendi tutte le chiavi delle decisioni dello scheduler
    decision_keys = [k for k in data.keys() if k.startswith("SchedulerDecision.")]

    # Trova il tempo massimo complessivo
    all_times = [t for key in decision_keys for t in data[key]]
    t_max = max(all_times) if all_times else 0

    plt.figure(figsize=(6, 6))

    for key in decision_keys:
        times = sorted(data[key])
        counts = list(range(1, len(times) + 1))

        # Se ci sono dati, aggiungi il tempo massimo per “prolungare” il conteggio
        if times:
            times = times + [t_max]
            counts = counts + [counts[-1]]

        plt.step(times, counts, label=key.split(".")[1], where='post')

    plt.xlabel("Tempo")
    plt.ylabel("Conteggio cumulativo")
    plt.title("Scelte dello Scheduler nel tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def main():
    # Sostituisci questo con il percorso effettivo del tuo file JSON
    json_file_path = "dqn_results/edge1.json"

    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Errore: File '{json_file_path}' non trovato.")
        exit()

    loss_plot(data)
    reward_plot(data)
    actions_plot(data)


if __name__ == "__main__":
    main()
