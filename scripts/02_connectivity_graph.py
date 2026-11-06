import mne
import yaml
import numpy as np
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

from mne_connectivity import spectral_connectivity_epochs
from prep_utils import map_triggers_to_conditions

# Включаем интерактивный режим для matplotlib (чтобы окна не зависали)
plt.ion() 
mne.set_log_level('ERROR')

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # Создаем папку results, если её нет
    os.makedirs("results", exist_ok=True)
    
    print("1. Загрузка конфигурации...")
    cfg = load_config()
    
    filepath = glob.glob(f"{cfg['data']['dataset_dir']}/**/*.set", recursive=True)[0]
    print(f"2. Обрабатываем файл: {filepath}")
    
    raw = mne.io.read_raw_eeglab(filepath, preload=True)
    raw.filter(l_freq=cfg['preprocessing']['l_freq'], h_freq=cfg['preprocessing']['h_freq'])
    
    print("3. Нарезка эпох...")
    new_events, final_event_id = map_triggers_to_conditions(raw, cfg['conditions']['target_events'])
    
    epochs = mne.Epochs(raw, new_events, final_event_id,
                        tmin=cfg['preprocessing']['tmin'], tmax=cfg['preprocessing']['tmax'],
                        baseline=(None, 0), reject=dict(eeg=cfg['preprocessing']['reject_uv'] * 1e-6), preload=True)

    print("4. Расчет связности (wPLI) для каждого условия...")
    conn_matrices = {}
    n_channels = len(epochs.ch_names)
    ch_names = epochs.ch_names
    
    for cond_name in epochs.event_id.keys():
        epochs_cond = epochs[cond_name]
        
        con = spectral_connectivity_epochs(
            epochs_cond, method=cfg['connectivity']['method'], mode='multitaper',
            fmin=cfg['connectivity']['fmin'], fmax=cfg['connectivity']['fmax'],
            faverage=True, n_jobs=1, verbose=False
        )
        
        W = con.get_data().squeeze()
        if W.shape != (n_channels, n_channels):
            W = W.reshape(n_channels, n_channels)
        
        # Делаем матрицу симметричной (на всякий случай)
        W = (W + W.T) / 2
        conn_matrices[cond_name] = W.copy()
        print(f"   - Матрица {W.shape} для {cond_name}")

    print("\n5. Сохранение матриц в CSV...")
    for cond_name, W in conn_matrices.items():
        df = pd.DataFrame(W, index=ch_names, columns=ch_names)
        filename = f"results/matrix_{cond_name}.csv"
        df.to_csv(filename)
        print(f"   - Сохранено: {filename}")

    print("\n6. Построение графов (MST) и расчет метрик...")
    
    # Создаем фигуру для визуализации всех 6 графов
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (cond_name, W) in enumerate(conn_matrices.items()):
        np.fill_diagonal(W, 0)
        
        W_inv = 1 - W
        W_inv[W_inv < 0] = 0
        
        # Создаем граф в networkx
        G = nx.from_numpy_array(W_inv)
        
        # Строим минимальное остовное дерево
        mst = nx.minimum_spanning_tree(G, weight='weight')
        
        n_edges = mst.number_of_edges()
        print(f"   Условие: {cond_name} | Узлов: {n_channels} | Ребер в MST: {n_edges}")

        # Визуализация графа
        ax = axes[i]
        # Располагаем узлы по кругу для красоты
        pos = nx.circular_layout(mst) 
        
        # Рисуем
        nx.draw_networkx_nodes(mst, pos, ax=ax, node_size=300, node_color='skyblue')
        nx.draw_networkx_edges(mst, pos, ax=ax, width=1.5, edge_color='gray')
        nx.draw_networkx_labels(mst, pos, ax=ax, font_size=8)
        
        ax.set_title(cond_name)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("results/all_graphs_mst.png", dpi=300)
    print("\n7. Графы сохранены в файл: results/all_graphs_mst.png")
    
    # Показываем график на экране (в VS Code он откроется во вкладке или отдельным окном)
    plt.show(block=True) 

if __name__ == "__main__":
    main()
