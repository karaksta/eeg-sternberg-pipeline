import mne
import yaml
import numpy as np
import glob
import pandas as pd
import networkx as nx
import os
from mne_connectivity import spectral_connectivity_epochs
from prep_utils import map_triggers_to_conditions

mne.set_log_level('ERROR')

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_graph_metrics(W):
    n_channels = W.shape[0]
    np.fill_diagonal(W, 0)
    W_inv = 1 - W
    W_inv[W_inv < 0] = 0
    G = nx.from_numpy_array(W_inv)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    ge = nx.global_efficiency(mst)
    cpl = nx.average_shortest_path_length(mst, weight='weight')
    degrees = [d for n, d in mst.degree()]
    leaf_fraction = degrees.count(1) / n_channels
    return {
        "global_efficiency": ge,
        "char_path_length": cpl,
        "leaf_fraction": leaf_fraction
    }

def main():
    cfg = load_config()
    
    # Ищем любой файл
    search_paths = [
        f"{cfg['data']['dataset_dir']}/**/*.set",
        f"{cfg['data']['dataset_dir']}/**/*.edf",
        f"{cfg['data']['dataset_dir']}/**/*.bdf"
    ]
    all_files = []
    for path in search_paths:
        all_files.extend(glob.glob(path, recursive=True))
        
    if not all_files:
        print("Файлы не найдены!")
        return
        
    filepath = sorted(all_files)[0]
    print(f"Обработка субъекта: {filepath}")
    
    if filepath.endswith(".set"):
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
    else:
        raw = mne.io.read_raw_edf(filepath, preload=True)
        
    raw.filter(l_freq=cfg['preprocessing']['l_freq'], h_freq=cfg['preprocessing']['h_freq'])
    
    new_events, final_event_id = map_triggers_to_conditions(raw, cfg['conditions']['target_events'])
    
    reject_val = cfg['preprocessing']['reject_uv'] * 1e-6
    epochs = mne.Epochs(raw, new_events, final_event_id,
                        tmin=cfg['preprocessing']['tmin'], tmax=cfg['preprocessing']['tmax'],
                        baseline=(None, 0), reject=dict(eeg=reject_val), preload=True)
    
    results = []
    print("Расчет метрик для 6 условий...")
    for cond_name in epochs.event_id.keys():
        con = spectral_connectivity_epochs(
            epochs[cond_name], method=cfg['connectivity']['method'], mode='multitaper',
            fmin=cfg['connectivity']['fmin'], fmax=cfg['connectivity']['fmax'],
            faverage=True, n_jobs=1, verbose=False
        )
        W = con.get_data().squeeze()
        if W.shape != (len(epochs.ch_names), len(epochs.ch_names)):
            W = W.reshape(len(epochs.ch_names), len(epochs.ch_names))
        
        metrics = compute_graph_metrics(W)
        metrics['condition'] = cond_name
        results.append(metrics)
        print(f"  {cond_name}: GE={metrics['global_efficiency']:.3f}, CPL={metrics['char_path_length']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv("results/sub_test_metrics.csv", index=False)
    print("\nМетрики сохранены в results/sub_test_metrics.csv")

if __name__ == "__main__":
    main()