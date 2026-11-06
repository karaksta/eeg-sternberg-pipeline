import mne
import glob
import pandas as pd
from collections import Counter
import os

# Отключаем лишние логи MNE, чтобы не засорять вывод
mne.set_log_level('WARNING') 

print("1. Старт скрипта...")
print("2. Ищу файлы ЭЭГ в папке v1...")

# Ищем только внутри папки v1
candidates = []
for ext in ["*.set", "*.edf", "*.bdf", "*.fif"]:
    candidates.extend(glob.glob(f"v1/**/{ext}", recursive=True))

if not candidates:
    print("❌ ОШИБКА: Файлы ЭЭГ не найдены внутри папки v1.")
    print("Пожалуйста, проверьте, какие именно файлы лежат в папке v1, и напишите мне их расширение.")
    exit()

print(f"✅ Найдено файлов: {len(candidates)}")
print(f"Беру для теста первый файл: {candidates[0]}\n")

f = candidates[0]

# Загружаем в зависимости от формата
try:
    if f.endswith(".set"):
        raw = mne.io.read_raw_eeglab(f, preload=False)
    elif f.endswith((".edf", ".bdf")):
        raw = mne.io.read_raw_edf(f, preload=False)
    elif f.endswith(".fif"):
        raw = mne.io.read_raw_fif(f, preload=False)
except Exception as e:
    print(f"❌ Ошибка при чтении файла: {e}")
    exit()

print("\n=== ИНФОРМАЦИЯ О ЗАПИСИ ===")
print(f"Количество каналов: {len(raw.ch_names)}")
print(f"Каналы: {raw.ch_names}")
print(f"Частота дискретизации: {raw.info['sfreq']} Гц")
print(f"Длительность: {raw.n_times / raw.info['sfreq']:.1f} секунд")

print("\n=== СОБЫТИЯ (ТРИГГЕРЫ) ===")
if len(raw.annotations):
    df = pd.DataFrame({
        "onset": raw.annotations.onset,
        "description": raw.annotations.description
    })
    print("Первые 15 событий:")
    print(df.head(15).to_string())
    
    print("\nУникальные коды триггеров (и их количество):")
    counts = Counter(raw.annotations.description)
    for code, count in counts.most_common():
        print(f"  {code}: {count} раз")
else:
    print("Внутри файла ЭЭГ аннотаций нет. Ищу отдельный файл events.tsv...")
    ev_files = glob.glob(os.path.join(os.path.dirname(f), "*events.tsv"))
    if ev_files:
        print(f"✅ Найден файл событий: {ev_files[0]}")
        d = pd.read_csv(ev_files[0], sep="\t")
        print(d.head(10).to_string())
    else:
        print("❌ Файл events.tsv тоже не найден.")

print("\n3. Скрипт успешно завершен.")