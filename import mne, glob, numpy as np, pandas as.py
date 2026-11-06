import mne, glob, numpy as np, pandas as pd
from collections import Counter

# --- 1. Найти файл первого субъекта ---
candidates = glob.glob("sub-01/**/*_eeg.set", recursive=True) + \
             glob.glob("sub-01/**/*_eeg.edf", recursive=True) + \
             glob.glob("sub-01/**/*_eeg.bdf", recursive=True)
print("Найдено файлов:", len(candidates))
for c in candidates[:5]:
    print("  ", c)

f = candidates[0]
raw = (mne.io.read_raw_eeglab(f, preload=False) if f.endswith(".set")
       else mne.io.read_raw_edf(f, preload=False))

print("\n=== INFO ===")
print(raw.info)
print("Каналы:", len(raw.ch_names), raw.ch_names)
print("sfreq:", raw.info["sfreq"], "Гц")
print("Длительность:", round(raw.n_times / raw.info["sfreq"], 1), "с")

# --- 2. События из аннотаций (EEGLAB) ---
print("\n=== ANNOTATIONS (первые 30) ===")
if len(raw.annotations):
    ann = raw.annotations
    df = pd.DataFrame({"onset": ann.onset,
                       "duration": ann.duration,
                       "description": ann.description})
    print(df.head(30).to_string())
    print("\nУникальные коды триггеров и их частоты:")
    print(Counter(ann.description).most_common())
else:
    print("Аннотаций нет — возможно, события в отдельном events.tsv")

# --- 3. Если есть events.tsv рядом — показать ---
for ev in glob.glob("sub-01/**/*_events.tsv", recursive=True):
    print("\n=== ", ev, "===")
    d = pd.read_csv(ev, sep="\t")
    print(d.head(15).to_string())
    print("\nКолонки:", list(d.columns))
    for col in ["trial_type", "value", "stim_file", "LOAD", "TASK"]:
        if col in d.columns:
            print(f"Уникальные {col}:", sorted(d[col].dropna().unique())[:20])
    break
