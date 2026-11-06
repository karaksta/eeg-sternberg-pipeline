import mne
import glob
import yaml
import os
import numpy as np

mne.set_log_level('ERROR')

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_first_subject_file(data_dir):
    candidates = glob.glob(f"{data_dir}/**/*.set", recursive=True)
    if not candidates:
        candidates = glob.glob(f"{data_dir}/**/*.edf", recursive=True)
    return candidates[0] if candidates else None


def main():
    print("1. Загрузка конфигурации...")
    cfg = load_config()
    
    print("2. Поиск файла данных...")
    filepath = get_first_subject_file(cfg['data']['dataset_dir'])
    if not filepath:
        print("Файлы не найдены!")
        return
    print(f"   Обрабатываем: {filepath}")

    print("3. Чтение данных...")
    if filepath.endswith(".set"):
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
    else:
        raw = mne.io.read_raw_edf(filepath, preload=True)
        
    print(f"   Частота дискретизации: {raw.info['sfreq']} Гц")

    print("4. Фильтрация сигнала...")
    raw.filter(l_freq=cfg['preprocessing']['l_freq'], 
               h_freq=cfg['preprocessing']['h_freq'], 
               fir_design='firwin')
    print("   Фильтрация завершена.")

    print("5. Создание эпох с группировкой условий...")
    new_events, final_event_id = map_triggers_to_conditions(raw, cfg['conditions']['target_events'])
    
    epochs = mne.Epochs(raw, new_events, final_event_id,
                        tmin=cfg['preprocessing']['tmin'],
                        tmax=cfg['preprocessing']['tmax'],
                        baseline=(None, 0),
                        reject=dict(eeg=cfg['preprocessing']['reject_uv'] * 1e-6),
                        preload=True)
    
    print("\n=== РЕЗУЛЬТАТ ЭПОХИРОВАНИЯ ===")
    print(f"Размерность данных: {epochs.get_data().shape}")
    
    print("\nСгруппированные условия (event_id):")
    for key in epochs.event_id.keys():
        count = len(epochs[key])
        print(f"  - {key}: {count} эпох")
        
    print("\nКонвейер успешно протестирован на одном субъекте!")

if __name__ == "__main__":
    main()
    