import numpy as np
import pandas as pd
from io import StringIO
import os
import matplotlib.pyplot as plt


# ============================================================
# 🔹 PARTE 1: Conversione CSV → JSON
# ============================================================
def convert_csv_to_json(input_path: str, output_path: str):
    """
    Converte un file CSV (anche con metadati InfluxDB) in un file JSON leggibile.
    """
    if not os.path.exists(input_path):
        print(f"❌ Errore: il file '{input_path}' non esiste.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Rimuove righe di metadati (quelle che iniziano con "#")
    data_lines = [line for line in lines if not line.startswith("#")]

    # Crea un DataFrame con i dati puliti
    clean_csv = StringIO("".join(data_lines))
    df = pd.read_csv(clean_csv)

    # Converte in JSON leggibile
    json_data = df.to_json(orient="records", indent=4, force_ascii=False)

    # Scrive il JSON su file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_data)

    print(f"✅ Conversione completata!\nFile JSON salvato come: {output_path}")


# ============================================================
# 🔹 PARTE 2: Grafico a gradini Scheduler (actions_plot)
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import json
import re


def actions_plot_from_json(json_path: str):
    """
    Legge un file JSON in formato InfluxDB (come 'dati_puliti.json') e
    costruisce un grafico cumulativo delle scelte dello scheduler
    per le azioni Exec, Edge, Cloud e Drop.
    """

    # --- 1️⃣ Legge il file JSON ---
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- 2️⃣ Prepara una mappa {azione: [timestamps]} ---
    actions = {"EXEC": [], "EDGE": [], "CLOUD": [], "DROP": []}

    for entry in data:
        name = str(entry.get("name", "")).upper()
        print(name)
        if name in actions:
            raw_value = entry.get("_value", "[]")
            # Estrae tutti i numeri dal testo della lista
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_value)
            actions[name].extend(map(float, numbers))

    # --- 3️⃣ Costruisce il grafico cumulativo ---
    plt.figure(figsize=(8, 6))

    t_max = max((max(v) for v in actions.values() if v), default=0)

    for name, times in actions.items():
        times = sorted(times)
        counts = list(range(1, len(times) + 1))

        if times:
            # “Prolunga” la curva fino al tempo massimo per continuità
            times = times + [t_max]
            counts = counts + [counts[-1]]

            plt.step(times, counts, label=name.capitalize(), where='post')

            # Etichetta il conteggio finale
            plt.text(t_max + 0.01 * (t_max or 1), counts[-1],
                     str(counts[-1]), verticalalignment='center')

    plt.xlabel("Tempo (s)")
    plt.ylabel("Conteggio cumulativo")
    plt.title("Scelte dello Scheduler nel tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def reward_plot(json_path, save_path="avg_reward_serverledge.npy"):
    # --- Carica i dati ---
    with open(json_path, "r") as f:
        data = json.load(f)

    # Cerca la voce che contiene i reward
    rewards = None
    for entry in data:
        if entry.get("name", "").lower() == "reward" or entry.get("_field", "") == "reward":
            val = entry.get("_value", None)
            if isinstance(val, str):
                # Converte la stringa JSON in lista
                rewards = json.loads(val.replace("'", '"'))
            elif isinstance(val, list):
                rewards = val
            break

    if rewards is None:
        raise ValueError("❌ Nessun campo 'reward' trovato nel file JSON.")

    # --- Calcolo media cumulativa ---
    rewards = np.array(rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)
    cumulative_avg = np.cumsum(rewards) / episodes

    print(f"Reward cumulativo: {np.sum(rewards)}")

    # --- Grafico ---
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, cumulative_avg, label='Media Cumulativa', color='blue', linewidth=1)
    plt.xlabel('Episodio')
    plt.ylabel('Reward medio')
    plt.title('Andamento del Reward nel tempo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    np.save(save_path, cumulative_avg)

def calcola_durata_media(json_path):
    # --- Carica i dati JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    duration_data = None

    # --- Cerca il campo 'Duration' ---
    for entry in data:
        if entry.get("name", "").lower() == "duration":
            raw_value = entry.get("_value", "")
            # Converti la stringa in lista di liste
            duration_data = json.loads(raw_value.replace("'", '"'))
            break

    if duration_data is None:
        raise ValueError("❌ Nessun campo 'Duration' trovato nel file JSON.")

    # --- Calcola media per ciascuna funzione ---
    durata_media = {}
    for i, lista in enumerate(duration_data, start=1):
        if lista:  # se non è vuota
            durata_media[f"f{i}"] = np.mean(lista)
        else:
            durata_media[f"f{i}"] = None  # nessun dato disponibile

    # --- Stampa risultati ---
    print("📊 Durata media per funzione:")
    for nome_funzione, durata in durata_media.items():
        if durata is not None:
            print(f"  {nome_funzione}: {durata:.4f} secondi")
        else:
            print(f"  {nome_funzione}: nessun dato disponibile")

    return durata_media


# ============================================================
# 🔹 MAIN
# ============================================================
def main():
    """
    Esegue la conversione e genera il grafico.
    """
    input_csv = "dqn_stats.csv"
    output_json = "dati_puliti.json"

    # Step 1: Conversione CSV → JSON
    convert_csv_to_json(input_csv, output_json)

    # Step 2: Grafico delle azioni dello scheduler
    #actions_plot_from_json(output_json)
    reward_plot(output_json)
    calcola_durata_media(output_json)



if __name__ == "__main__":
    main()
