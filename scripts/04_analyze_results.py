import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import os
import warnings

warnings.filterwarnings("ignore") 

sns.set_theme(style="whitegrid", context="talk")

def main():
    filepath = "results/all_subjects_metrics.csv"
    if not os.path.exists(filepath):
        print(f"Файл {filepath} не найден. Сначала запустите run_all.py!")
        return

    print("1. Загрузка данных...")
    df = pd.read_csv(filepath)
    df = df[df['condition'] != 'ERROR'].copy()
    
    df['load'] = df['condition'].apply(lambda x: int(x.split('_')[1]))
    df['task'] = df['condition'].apply(lambda x: x.split('_')[2])
    
    print(f"Загружено {len(df)} записей от {df['subject'].nunique()} субъектов.")

    print("\n2. Построение графиков...")
    metrics = ['global_efficiency', 'char_path_length', 'leaf_fraction']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        sns.boxplot(
            data=df, 
            x='load', 
            y=metric, 
            hue='task',
            palette=['#4C72B0', '#C44E52'],
            ax=axes[i],
            showfliers=False
        )
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")
        axes[i].set_xlabel("Load (Memory items)")
        axes[i].set_ylabel("")
        
        if i == 0:
            axes[i].legend(title="Task")
        else:
            axes[i].get_legend().remove()

    plt.tight_layout()
    plot_path = "results/plots_metrics_by_condition.png"
    plt.savefig(plot_path, dpi=300)
    plt.close(fig) # Закрываем фигуру для очистки памяти
    print(f"   Графики сохранены: {plot_path}")

    print("\n3. Статистический анализ (Two-way Repeated Measures ANOVA)...")
    print("=" * 60)
    
    for metric in metrics:
        print(f"\nМетрика: {metric.replace('_', ' ').title()}")
        print("-" * 60)
        
        try:
            aov = pg.rm_anova(data=df, dv=metric, within=['load', 'task'], subject='subject', detailed=True)
            
            for index, row in aov.iterrows():
                source = row['Source']
                p_val = row['p-unc']
                f_val = row['F']
                
                if pd.isna(p_val):
                    sig = ""
                else:
                    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns (не значимо)"))
                
                print(f"  {source:15}: F = {f_val:6.2f}, p = {p_val:.4f} [{sig}]")
                
            print("\n  Средние значения (Mean ± Std):")
            means = df.groupby(['task', 'load'])[metric].agg(['mean', 'std'])
            for (task, load), row in means.iterrows():
                print(f"    Task: {task:12} | Load: {load} | {row['mean']:.4f} ± {row['std']:.4f}")
                
        except Exception as e:
            print(f"  Ошибка при расчете ANOVA: {e}")
            
        print("=" * 60)

    print("\n4. Готово! Откройте картинку из папки results/ чтобы увидеть результаты.")

if __name__ == "__main__":
    main()
    