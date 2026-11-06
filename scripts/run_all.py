import mne
import yaml
import numpy as np
import glob
import pandas as pd
import networkx as nx
import os
import time
import traceback
from joblib import Parallel, delayed
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

def process_single_subject(filepath, cfg):
    subject_id = os.path.basename(filepath).split('_')[0]
    try:
        # Универсальная загрузка под любой формат
        if filepath.endswith(".set"):
            raw = mne.io.read_raw_eeglab(filepath, preload=True)
        elif filepath.endswith((".edf", ".bdf")):
            raw = mne.io.read_raw_edf(filepath, preload=True)
        elif filepath.endswith(".fif"):
            raw = mne.io.read_raw_fif(filepath, preload=True)
        
        raw.filter(l_freq=cfg['preprocessing']['l_freq'], h_freq=cfg['preprocessing']['h_freq'])
        
        new_events, final_event_id = map_triggers_to_conditions(raw, cfg['conditions']['target_events'])
        
        if len(final_event_id) == 0:
            return [{"subject": subject_id, "condition": "ERROR", "error": "No valid triggers"}]

        reject_val = cfg['preprocessing']['reject_uv'] * 1e-6
        
        epochs = mne.Epochs(raw, new_events, final_event_id,
                            tmin=cfg['preprocessing']['tmin'], tmax=cfg['preprocessing']['tmax'],
                            baseline=(None, 0), reject=dict(eeg=reject_val), preload=True)
        
        subject_results = []
        
        for cond_name in epochs.event_id.keys():
            if len(epochs[cond_name]) < 5:
                continue
                
            con = spectral_connectivity_epochs(
                epochs[cond_name], method=cfg['connectivity']['method'], mode='multitaper',
                fmin=cfg['connectivity']['fmin'], fmax=cfg['connectivity']['fmax'],
                faverage=True, n_jobs=1, verbose=False
            )
            W = con.get_data().squeeze()
            if W.shape != (len(epochs.ch_names), len(epochs.ch_names)):
                W = W.reshape(len(epochs.ch_names), len(epochs.ch_names))
            
            metrics = compute_graph_metrics(W)
            metrics['subject'] = subject_id
            metrics['condition'] = cond_name
            metrics['error'] = None
            subject_results.append(metrics)
            
        return subject_results
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return [{"subject": subject_id, "condition": "ERROR", "error": error_msg}]

def main():
    os.makedirs("results", exist_ok=True)
    cfg = load_config()
    
    print("1. Поиск всех файлов субъектов...")
    # Ищем файлы ЛЮБОГО популярного формата ЭЭГ
    search_paths = [
        f"{cfg['data']['dataset_dir']}/**/*.set",
        f"{cfg['data']['dataset_dir']}/**/*.edf",
        f"{cfg['data']['dataset_dir']}/**/*.bdf",
        f"{cfg['data']['dataset_dir']}/**/*.fif"
    ]
    all_files = []
    for path in search_paths:
        all_files.extend(glob.glob(path, recursive=True))
    
    all_files = sorted(list(set(all_files))) # Убираем дубликаты
    
    if not all_files:
        print(f"❌ ОШИБКА: Не найдено ни одного файла в папке {cfg['data']['dataset_dir']}")
        return

    total_subjects = len(all_files)
    print(f"   Найдено файлов: {total_subjects}")
    
    print("\n2. Запуск параллельной обработки...")
    start_time = time.time()
    
    results_nested = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_single_subject)(f, cfg) for f in all_files
    )
    
    end_time = time.time()
    print(f"\n3. Обработка завершена за {(end_time - start_time) / 60:.2f} минут.")
    
    print("\n4. Сбор результатов в единую таблицу...")
    flat_results = [item for sublist in results_nested for item in sublist]
    
    if not flat_results:
        print("❌ ОШИБКА: Данных для сохранения нет.")
        return

    df = pd.DataFrame(flat_results)
    df = df.sort_values(by=['subject', 'condition'])
    
    output_filename = "results/all_subjects_metrics.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"   Таблица сохранена: {output_filename}")
    print(f"   Всего строк: {len(df)}")
    
    errors = df[df['condition'] == 'ERROR']
    if not errors.empty:
        print(f"\n⚠️ ВНИМАНИЕ: Ошибки у {len(errors)} субъектов (подробности в CSV).")
    else:
        print("\n✅ Все субъекты обработаны без ошибок!")

if __name__ == "__main__":
    main()
    