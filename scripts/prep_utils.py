import mne
import numpy as np

def map_triggers_to_conditions(raw, target_events):
    """Преобразует сырые триггеры в чистые условия (load_5_manipulation и т.д.)"""
    events, event_id_raw = mne.events_from_annotations(raw)
    new_event_id = {name: [] for name in target_events.keys()}
    new_events = []
    
    for event, raw_id in zip(events, events[:, 2]):
        # Получаем строку-описание триггера (например, '250')
        desc = [k for k, v in event_id_raw.items() if v == raw_id][0]
        
        # Если код состоит из 3 цифр (например 250)
        if len(desc) >= 3 and desc.isdigit():
            load = int(desc[-2])
            task = desc[-1]
            
            for cond_name, mapping in target_events.items():
                if mapping['load'] == load and mapping['task'] == task:
                    if cond_name not in new_event_id:
                        new_event_id[cond_name] = []
                    new_event_id[cond_name].append(len(new_events))
                    # Заменяем сырой ID на новый индекс условия
                    new_events.append([event[0], event[1], list(target_events.keys()).index(cond_name) + 1])
                    break
                    
    new_events = np.array(new_events)
    final_event_id = {name: i+1 for i, name in enumerate(target_events.keys()) if len(new_event_id[name]) > 0}
    
    return new_events, final_event_id
