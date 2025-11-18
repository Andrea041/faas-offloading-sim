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


def plot_qos_piecharts(json_path: str):
    """
    Estrae i valori di utilità, penalità tempo e penalità scarto
    per le classi QoS (UPCritical2, UPCritical1, UPStandard, UPBatch)
    e genera tre grafici a torta, nascondendo i valori zero e mostrando
    i valori assoluti invece delle percentuali.
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    qos_targets = ["UPCritical2", "UPCritical1", "UPStandard", "UPBatch"]

    utilita = {}
    pen_tempo = {}
    pen_scarto = {}

    for entry in data:
        name = entry.get("name", "")
        if name in qos_targets:
            raw = entry.get("_value", "")
            try:
                values = json.loads(raw.replace("'", '"')) if isinstance(raw, str) else raw
            except Exception:
                continue

            if isinstance(values, list) and len(values) >= 3:
                utilita[name] = float(values[0])
                pen_tempo[name] = float(values[1])
                pen_scarto[name] = float(values[2])

    if not utilita:
        print("⚠️ Nessun campo QoS trovato nel file JSON.")
        return

    def plot_single_pie(values: dict, title: str):
        # Rimuove voci con valore zero
        filtered = {k: v for k, v in values.items() if v != 0}
        if not filtered:
            print(f"⚠️ Nessun valore non nullo per: {title}")
            return

        labels = list(filtered.keys())
        sizes = list(filtered.values())

        # 🎨 Colori QoS coerenti e gradevoli
        color_map = {
            "UPCritical2": "#E74C3C",  # rosso
            "UPCritical1": "#3498DB",  # blu
            "UPStandard": "green",  # verde
            "UPBatch": "#9B59B6"  # viola
        }

        label_map = {
            "UPCritical2": "Critical-2",  # rosso
            "UPCritical1": "Critical-1",  # blu
            "UPStandard": "Standard",  # verde
            "UPBatch": "Batch"  # viola
        }

        colors = [color_map.get(lbl, "#7f8c8d") for lbl in labels]
        etichette = [label_map.get(lbl) for lbl in labels]

        # Esplosione della fetta più grande per leggibilità
        explode = [0.02 for _ in sizes]  # tutte distanziate allo stesso modo

        # Funzione per mostrare valore assoluto nella fetta
        def fmt(val_percent):
            total = sum(sizes)
            value = int(round(val_percent * total / 100.0))
            return f"{value}"

        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        fig.suptitle(title, x=0.5, y=0.97, ha="center")

        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=etichette,
            colors=colors,
            explode=explode,
            startangle=140,
            autopct=fmt,
            pctdistance=0.6,
            labeldistance=1.12,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )

        # Centrare davvero il grafico nella figura
        ax.set_aspect("equal")
        # pos = [left, bottom, width, height]
        ax.set_position([0.08, 0.08, 0.84, 0.84])

        # Migliora leggibilità dei valori interni
        for txt in autotexts:
            txt.set_color("white")
            #txt.set_weight("bold")
            txt.set_fontsize(11)

        plt.tight_layout()
        plt.show()

    # 🔥 Generazione grafici aggiornati
    plot_single_pie(utilita, "")
    plot_single_pie(pen_tempo, "")
    plot_single_pie(pen_scarto, "")



# ============================================================
# 🔹 PARTE 2: Grafico a gradini Scheduler (actions_plot)
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import json
import re


import json
import re
import matplotlib.pyplot as plt

def actions_plot_from_json(json_path: str):
    """
    Legge un file JSON in formato InfluxDB e costruisce un grafico cumulativo
    delle scelte dello scheduler per le azioni Exec, Edge, Cloud e Drop.
    La label finale del CLOUD viene abbassata visivamente per evitare sovrapposizioni.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions = {"EXEC": [], "EDGE": [], "CLOUD": [], "DROP": []}

    for entry in data:
        name = str(entry.get("name", "")).upper()
        if name in actions:
            raw_value = entry.get("_value", "[]")
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_value)
            actions[name].extend(map(float, numbers))

    fig, ax = plt.subplots(figsize=(6, 6))

    # calcola t_max dai timestamps (0 se vuoto)
    t_max = max((max(v) for v in actions.values() if v), default=0)

    for name, times in actions.items():
        times = sorted(times)
        if not times:
            continue

        # costruisco la curva step con punto iniziale (0) per continuità
        times_plot = [0.0] + times
        counts = [0] + list(range(1, len(times_plot)))

        if name == "EXEC":
            label = "Esecuzione locale"
            color = "#1f77b4"
        elif name == "EDGE":
            label = "Offload Edge"
            color = "#2ca02c"
        elif name == "CLOUD":
            label = "Offload Cloud"
            color = "#ff7f0e"
        else:
            label = "Drop"
            color = "#d62728"

        ax.step(times_plot, counts, label=label, where='post', color=color)

        # posizione dati della label: in corrispondenza dell'ultimo punto (t_max)
        x_text = t_max
        y_text = counts[-1]

        # se vuoi che il testo stia leggermente a destra del grafico:
        x_offset_points = 6  # spostamento orizzontale in punti (display)
        # solo per CLOUD applichiamo uno spostamento verticale in punti (negativo = verso il basso)
        if name == "CLOUD":
            y_offset_points = -6   # prova -15, -25, -35 per aumentare lo spostamento
        else:
            y_offset_points = 0

        # Annotate con offset in punti: mantiene lo spostamento visivo indipendente dalla scala dati
        ax.annotate(
            str(counts[-1]),
            xy=(x_text, y_text),                 # punto di ancoraggio in dati
            xytext=(x_offset_points, y_offset_points),  # offset in punti
            textcoords='offset points',
            ha='left',
            va='center',
            bbox=dict(facecolor='white', alpha=0, edgecolor='none'),
            clip_on=False
        )

    # Assicuriamoci che l'area x includa lo spazio per le label a destra
    x_margin = (t_max or 1) * 0.06  # 6% di margine a destra
    ax.set_xlim(left=0, right=t_max + x_margin)

    ax.set_xlabel("Tempo (sec)")
    ax.set_ylabel("Conteggio cumulativo per azione")
    ax.set_title("Azioni eseguite nel tempo")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
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
    rewards = rewards[1:]
    rewards = np.array(rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)
    cumulative_avg = np.cumsum(rewards) / episodes

    # --- Aggiunge il punto iniziale (0, 0) per partire dall’origine ---
    episodes = np.insert(episodes, 0, 0)
    cumulative_avg = np.insert(cumulative_avg, 0, 0.0)

    print(f"Reward cumulativo: {np.sum(rewards)}")

    # --- Grafico ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, cumulative_avg, label='Reward medio cumulativo', linewidth=1)
    plt.xlabel('Numero di richieste elaborate')
    plt.ylabel('Reward')
    plt.title('Andamento del reward medio')
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

def calcola_cost_media(json_path):
    # --- Carica i dati JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    duration_data = None

    # --- Cerca il campo 'OffloadLatencyCloud' ---
    for entry in data:
        if entry.get("name", "").lower() == "cost":
            raw_value = entry.get("_value", "")

            # Se è già una lista, non serve json.loads
            if isinstance(raw_value, list):
                duration_data = raw_value
            else:
                try:
                    duration_data = json.loads(str(raw_value).replace("'", '"'))
                except json.JSONDecodeError:
                    raise ValueError("⚠️ Il campo '_value' non è in formato JSON valido.")

            break

    if duration_data is None:
        raise ValueError("❌ Nessun campo 'Cost' trovato nel file JSON.")

    # --- Conversione a numeri e calcolo media ---
    flat_data = [item for sublist in duration_data for item in (sublist if isinstance(sublist, list) else [sublist])]
    # Gestione caso: lista di liste o lista piatta
    numeric_data = [float(x) for x in flat_data if str(x).strip() not in ("", "None")]

    if not numeric_data:
        raise ValueError("⚠️ Nessun valore numerico valido trovato in 'Cost'.")

    sumval = np.sum(numeric_data)

    # --- Stampa risultati ---
    print("📊 Costo totale:")
    print(f"Costo totale dell'esecuzione {sumval:.4f}")

def calcola_penalty_tempo_media(json_path):
    # --- Carica i dati JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)


    # --- Cerca il campo 'OffloadLatencyCloud' ---
    for entry in data:
        if entry.get("name", "").lower() == "deadlinepenalty":
            raw_value = entry.get("_value", "")

            # Se è già una lista, non serve json.loads
            if isinstance(raw_value, list):
                duration_data = raw_value
            else:
                try:
                    duration_data = json.loads(str(raw_value).replace("'", '"'))
                except json.JSONDecodeError:
                    raise ValueError("⚠️ Il campo '_value' non è in formato JSON valido.")

            break
    if duration_data is None:
        raise ValueError("❌ Nessun campo 'deadline penalty' trovato nel file JSON.")

    # --- Conversione a numeri e calcolo media ---
    flat_data = [item for sublist in duration_data for item in (sublist if isinstance(sublist, list) else [sublist])]
    # Gestione caso: lista di liste o lista piatta
    numeric_data = []
    for x in flat_data:
        try:
            numeric_data.append(float(x))
        except (ValueError, TypeError):
            continue

    if not numeric_data:
        raise ValueError("⚠️ Nessun valore numerico valido trovato in 'Deadline penalty'")

    sumval = np.sum(numeric_data)

    # --- Stampa risultati ---
    print("📊 Penalità di tempo totale:")
    print(f"Penalità di tempo totale dell'esecuzione {sumval:.4f}")


def calcola_init_media(json_path):
    # --- Carica i dati JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    duration_data = None

    # --- Cerca il campo 'Duration' ---
    for entry in data:
        if entry.get("name", "").lower() == "inittime":
            raw_value = entry.get("_value", "")
            # Converti la stringa in lista di liste
            duration_data = json.loads(raw_value.replace("'", '"'))
            break

    if duration_data is None:
        raise ValueError("❌ Nessun campo 'InitTime' trovato nel file JSON.")

    # --- Calcola media per ciascuna funzione ---
    durata_media = {}
    for i, lista in enumerate(duration_data, start=1):
        if lista:  # se non è vuota
            durata_media[f"f{i}"] = np.mean(lista)
        else:
            durata_media[f"f{i}"] = None  # nessun dato disponibile

    # --- Stampa risultati ---
    print("📊 Init medio per funzione:")
    for nome_funzione, durata in durata_media.items():
        if durata is not None:
            print(f"  {nome_funzione}: {durata:.4f} secondi")
        else:
            print(f"  {nome_funzione}: nessun dato disponibile")

    return durata_media

def calcola_offC_media(json_path):
    # --- Carica i dati JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    duration_data = None

    # --- Cerca il campo 'OffloadLatencyCloud' ---
    for entry in data:
        if entry.get("name", "").lower() == "offloadlatencycloud":
            raw_value = entry.get("_value", "")

            # Se è già una lista, non serve json.loads
            if isinstance(raw_value, list):
                duration_data = raw_value
            else:
                try:
                    duration_data = json.loads(str(raw_value).replace("'", '"'))
                except json.JSONDecodeError:
                    raise ValueError("⚠️ Il campo '_value' non è in formato JSON valido.")

            break

    if duration_data is None:
        raise ValueError("❌ Nessun campo 'OffloadLatencyCloud' trovato nel file JSON.")

    # --- Conversione a numeri e calcolo media ---
    # Gestione caso: lista di liste o lista piatta
    flat_data = [item for sublist in duration_data for item in (sublist if isinstance(sublist, list) else [sublist])]
    numeric_data = [float(x) for x in flat_data if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()]

    if not numeric_data:
        raise ValueError("⚠️ Nessun valore numerico valido trovato in 'OffloadLatencyCloud'.")

    mean_value = np.mean(numeric_data)

    # --- Stampa risultati ---
    print("📊 Offload edge per funzione:")
    print(f"Durata media per offload al cloud: {mean_value:.4f} secondi")


def calcola_offE_media(json_path):
    # --- Carica i dati JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    duration_data = None

    # --- Cerca il campo 'OffloadLatencyEdge' ---
    for entry in data:
        if entry.get("name", "").lower() == "offloadlatencyedge":
            raw_value = entry.get("_value", "")

            # Se è già una lista, non serve json.loads
            if isinstance(raw_value, list):
                duration_data = raw_value
            else:
                try:
                    duration_data = json.loads(str(raw_value).replace("'", '"'))
                except json.JSONDecodeError:
                    raise ValueError("⚠️ Il campo '_value' non è in formato JSON valido.")

            break

    if duration_data is None:
        raise ValueError("❌ Nessun campo 'OffloadLatencyEdge' trovato nel file JSON.")

    # --- Conversione a numeri e calcolo media ---
    # Gestione caso: lista di liste o lista piatta
    flat_data = [item for sublist in duration_data for item in (sublist if isinstance(sublist, list) else [sublist])]
    numeric_data = [float(x) for x in flat_data if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()]

    if not numeric_data:
        raise ValueError("⚠️ Nessun valore numerico valido trovato in 'OffloadLatencyEdge'.")

    mean_value = np.mean(numeric_data)

    # --- Stampa risultati ---
    print("📊 Offload edge per funzione:")
    print(f"Durata media per offload all'edge: {mean_value:.4f} secondi")


# ============================================================
# 🔹 MAIN
# ============================================================
def main():
    """
    Esegue la conversione e genera il grafico.
    """
    input_csv = "dati.csv"
    output_json = "try.json"

    # Step 1: Conversione CSV → JSON
    convert_csv_to_json(input_csv, output_json)

    # Step 2: Grafico delle azioni dello scheduler
    actions_plot_from_json(output_json)
    reward_plot(output_json)
    calcola_durata_media(output_json)
    calcola_offC_media(output_json)
    calcola_offE_media(output_json)
    calcola_init_media(output_json)
    calcola_cost_media(output_json)
    calcola_penalty_tempo_media(output_json)
    plot_qos_piecharts(output_json)


if __name__ == "__main__":
    main()
