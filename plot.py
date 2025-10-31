import json

TRANSFER = False

def loss_plot(data, epsilon=1e-3):
    if "loss" in data:
        loss_values = np.array(data["loss"])
    else:
        print("Errore: La chiave 'loss' non è presente nel file JSON.")
        return

    iterations = np.arange(1, len(loss_values) + 1)

    # Calcolo media cumulativa
    cumulative_avg_loss = np.cumsum(loss_values) / iterations

    # Valore di riferimento (valore medio finale)
    final_mean = cumulative_avg_loss[-1]

    # Limiti del tubo di convergenza
    upper_bound = final_mean + epsilon
    lower_bound = final_mean - epsilon

    # Identificazione della prima iterazione in cui la curva entra nel tubo e ci resta
    inside_tube = (cumulative_avg_loss >= lower_bound) & (cumulative_avg_loss <= upper_bound)

    converged_idx = None
    for i in range(len(inside_tube)):
        # Verifica se da questo punto in poi rimane sempre dentro il tubo
        if inside_tube[i:].all():
            converged_idx = i
            break

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss_values, label="Loss", alpha=0.4)
    plt.plot(iterations, cumulative_avg_loss, label="Loss Media Cumulativa", color='red', linewidth=1)

    # Disegna il tubo di convergenza
    plt.fill_between(iterations, lower_bound, upper_bound, color='green', alpha=0.3, label=f"Ampiezza banda di convergenza: ±{epsilon:.3g}")

    # Indica il punto in cui entra nel tubo
    if converged_idx is not None:
        plt.axvline(iterations[converged_idx], color='black', linestyle='--', alpha=0.4)
        plt.text(iterations[converged_idx], upper_bound,
                 f"Convergenza a partire dalla iterazione {iterations[converged_idx]}",
                 color='black', va='bottom', ha='right', fontsize=9)

    plt.title("Loss con Media Cumulativa")
    plt.xlabel("Iterazioni")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def reward_plot(data, save_path="avg_reward.npy"):
    # Estrae i reward
    rewards = data["reward"]

    if not rewards:
        raise ValueError("Nessun campo 'reward' trovato nel file JSON.")

    episodes = np.arange(1, len(rewards) + 1)

    # Calcolo media cumulativa
    cumulative_avg = np.cumsum(rewards) / episodes

    # Grafico
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, cumulative_avg, label='Media Cumulativa', color='blue', linewidth=1)
    plt.xlabel('Episodio')
    plt.ylabel('Reward')
    plt.title('Andamento Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    np.save(save_path, cumulative_avg)

def reward_plot_transfer(data, epsilon=1e-2, adaptive=False, k=1.0, n_tail=50, save_path="avg_reward_tl.npy"):
    # Controlli e preparazione
    if "reward" not in data:
        raise ValueError("Errore: la chiave 'reward' non è presente nel dict 'data'.")
    rewards = np.array(data["reward"], dtype=float)
    if rewards.size == 0:
        raise ValueError("Errore: 'reward' è vuoto.")

    episodes = np.arange(1, len(rewards) + 1)

    # Media cumulativa
    cumulative_avg = np.cumsum(rewards) / episodes

    print(f"Reward cumulativo: {np.sum(rewards)}")

    # Calcolo epsilon adattivo se richiesto
    if adaptive:
        tail = rewards[-n_tail:] if len(rewards) >= n_tail else rewards
        if tail.size == 0:
            raise ValueError("Impossibile calcolare epsilon adattivo: non ci sono reward sufficienti.")
        epsilon = k * np.std(tail)

    # Valori del tubo centrati sul valore limite (media cumulativa finale)
    final_mean = cumulative_avg[-1]
    upper_bound = final_mean + epsilon
    lower_bound = final_mean - epsilon

    # Verifica quali punti della media cumulativa sono dentro il tubo
    inside_tube = (cumulative_avg >= lower_bound) & (cumulative_avg <= upper_bound)

    # Trova la prima iterazione da cui la media cumulativa rimane sempre dentro il tubo
    converged_idx = None
    for i in range(len(inside_tube)):
        if inside_tube[i:].all():
            converged_idx = i  # indice 0-based; corrisponde a episodes[i]
            break

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, cumulative_avg, label='Media Cumulativa', linewidth=1)
    plt.fill_between(episodes, lower_bound, upper_bound, alpha=0.3,
                     label=f"Ampiezza banda di convergenza: ±{epsilon:.3g}")

    # Se trovato, disegna linea verticale e annotazione (episodi sono 1-based)
    if converged_idx is not None:
        x_conv = episodes[converged_idx]
        plt.axvline(x_conv, color='black', linestyle='--', alpha=0.8)
        # posiziono il testo sopra il tubo
        y_text = upper_bound + 0.02 * (np.max(cumulative_avg) - np.min(cumulative_avg) + 1e-8)
        plt.text(x_conv, y_text, f"Episodio {int(x_conv)}", color='black',
                 va='bottom', ha='left', fontsize=9)

    plt.xlabel('Episodio')
    plt.ylabel('Reward')
    plt.title('Andamento Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    np.save(save_path, cumulative_avg)


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

        # Stampa il conteggio finale a lato della curva
        if counts:
            plt.text(t_max + 0.01*(t_max or 1), counts[-1], str(counts[-1]),
                     verticalalignment='center')

    plt.xlabel("Tempo")
    plt.ylabel("Conteggio cumulativo")
    plt.title("Scelte dello Scheduler nel tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

def plot_transfer_performance(file_no_transfer,
                              file_transfer,
                              asymptotic_window=10,
                              jumpstart_zoom_window=2,
                              smooth=False,
                              savepath=None,
                              figsize=(12,6)):
    """
    Plotta le curve 'Transfer' e 'No Transfer' leggendo due .npy,
    taglia gli zeri iniziali mantenendo gli indici originali e calcola:
      - Jumpstart (al primo indice comune)
      - Time to Threshold (diferenza degli indici originali)
      - Asymptotic performance (media degli ultimi asymptotic_window punti)
    Argomenti:
      file_no_transfer : str percorso .npy (no TL)
      file_transfer    : str percorso .npy (TL)
      threshold        : float o None (se None usa 95° percentile di no_transfer_trimmed)
      asymptotic_window: int numero di punti finali per media asintotica
      smooth           : Bool se True applica una semplice media mobile per togliere rumore
      savepath         : str o None, salva immagine se specificato
      figsize          : tuple, dimensione figura matplotlib
    Ritorna:
      dict con metriche calcolate
    """

    # --- Carica dati ---
    r_no = np.load(file_no_transfer)
    r_tl = np.load(file_transfer)

    # --- Trova primo indice non zero (manteniamo INDICI ORIGINALI) ---
    def first_nonzero_index(arr):
        nz = np.where(arr > 0)[0]
        return int(nz[0]) if nz.size > 0 else 0

    i0_no = first_nonzero_index(r_no)
    i0_tl = first_nonzero_index(r_tl)

    r_no_trim = r_no[i0_no:]
    r_tl_trim = r_tl[i0_tl:]

    # indici originali per il plotting
    x_no = np.arange(i0_no, i0_no + len(r_no_trim))
    x_tl = np.arange(i0_tl, i0_tl + len(r_tl_trim))

    # --- Opzionale smoothing (media mobile) ---
    if smooth:
        def movavg(a, k=5):
            if k <= 1:
                return a
            return np.convolve(a, np.ones(k)/k, mode='same')
        r_no_plot = movavg(r_no_trim, k=5)
        r_tl_plot = movavg(r_tl_trim, k=5)
    else:
        r_no_plot = r_no_trim
        r_tl_plot = r_tl_trim

    # --- Jumpstart: prendi il primo indice comune (max dei due start) ---
    jumpstart = r_tl_trim[0] - r_no_trim[0]

    # --- Asymptotic performance: media ultimi asymptotic_window punti (se disponibili) ---
    def asymptotic_mean(arr, window):
        if len(arr) == 0:
            return np.nan
        w = min(window, len(arr))
        return float(np.mean(arr[-w:]))

    asymp_no = asymptotic_mean(r_no_trim, asymptotic_window)
    asymp_tl = asymptotic_mean(r_tl_trim, asymptotic_window)
    asymp_diff = asymp_tl - asymp_no

    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.plot(x_tl[:len(x_no)], r_tl_plot[:len(r_no_plot)], label='Transfer', linewidth=1)
    plt.plot(x_no, r_no_plot, label='No Transfer', linewidth=1)

    # Asymptotic performance: freccia verticale alla fine della curva TL
    xt_end = min(x_tl[-1], x_no[-1])
    plt.annotate('', xy=(xt_end, asymp_no), xytext=(xt_end, asymp_tl),
                 arrowprops=dict(arrowstyle='<->', lw=1.2))
    plt.text(xt_end + 0.02*(plt.xlim()[1]-plt.xlim()[0]),
             (asymp_no+asymp_tl)/2, 'Asymptotic\nPerformance', va='center', ha='left', fontsize=10)

    plt.xlabel('Episodio')
    plt.ylabel('Reward')
    plt.title('Asymptotic Performance')
    plt.legend()
    plt.grid(True)

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()

    # === PLOT ZOOMATO SUL JUMPSTART ===
    plt.figure(figsize=(8, 5))

    # Reindicizza da 0 solo per il confronto iniziale
    r_no_zoom = r_no_trim[:jumpstart_zoom_window]
    r_tl_zoom = r_tl_trim[:jumpstart_zoom_window]
    x_zoom = np.arange(len(r_no_zoom))  # reindicizzato da 0
    print(r_no_zoom, r_tl_zoom)

    plt.plot(x_zoom, r_tl_zoom[:len(x_zoom)], label='Transfer', linewidth=1)
    plt.plot(x_zoom, r_no_zoom[:len(x_zoom)], label='No Transfer', linewidth=1)

    # Disegna freccia jumpstart
    plt.annotate('', xy=(0, r_no_zoom[0]), xytext=(0, r_tl_zoom[0]),
                 arrowprops=dict(arrowstyle='<->', lw=1.2, color='black'))

    y_label = r_tl_zoom[0] * 1.2
    plt.text(0.025, y_label,
             f'Jumpstart = {jumpstart:.6f}',
             color='black', va='bottom', ha='left', fontsize=10)

    plt.title('Zoom sul Jumpstart')
    plt.xlabel('Episodio')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.yscale('log')
    plt.title('Jumpstart (scala logaritmica)')
    plt.show()

    # --- Transfer Ratio normalizzato ---
    min_len = min(len(r_no_trim), len(r_tl_trim))
    area_no = np.trapz(r_no_trim[:min_len])
    area_tl = np.trapz(r_tl_trim[:min_len])
    transfer_ratio = (area_tl - area_no) / area_no if area_no != 0 else np.nan

    # --- Ritorna le metriche per uso programmato ---
    metrics = {
        'start_index_no': int(i0_no),
        'start_index_tl': int(i0_tl),
        'jumpstart': float(jumpstart) if not np.isnan(jumpstart) else None,
        'asymp_no': float(asymp_no),
        'asymp_tl': float(asymp_tl),
        'asymp_diff': float(asymp_diff),
        'area_no': float(area_no),
        'area_tl': float(area_tl),
        'transfer_ratio': float(transfer_ratio),
    }

    # stampa compatta
    print("=== METRICHE ===")
    print(f"Start index (No TL): {metrics['start_index_no']}")
    print(f"Start index (TL):    {metrics['start_index_tl']}")
    print(f"Valore No Transfer = {r_no_trim[0]:.4f}")
    print(f"Valore Transfer    = {r_tl_trim[0]:.4f}")
    print(f"Jumpstart: {metrics['jumpstart']:.4f}")
    print(f"Asymptotic TL: {metrics['asymp_tl']:.4f}, Asymptotic No: {metrics['asymp_no']:.4f}, diff: {metrics['asymp_diff']:.4f}"),
    print(f"Area no transfer: {metrics['area_no']:.4f}"),
    print(f"Area transfer: {metrics['area_tl']:.4f}"),
    print(f"Transfer Ratio: {metrics['transfer_ratio']:.4f}")




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

    if TRANSFER:
        reward_plot_transfer(data)
        plot_transfer_performance("avg_reward_serverledge.npy", "avg_reward_tl.npy")
    else:
        reward_plot(data)

    actions_plot(data)


if __name__ == "__main__":
    main()
