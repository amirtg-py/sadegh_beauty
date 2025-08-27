import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --- 1. تنظیمات اولیه: مسیر کامل پوشه‌ها ---
# لطفاً مطمئن شوید این مسیرها دقیقاً با ساختار پوشه شما مطابقت دارند
GENERATED_DATA_DIR = r'D:\work_station\radar\final\data_generated_final_fixed'
REAL_DATA_DIR = r'D:\work_station\radar_co\radar\final\data2\project_files\lstm_out\real' # مسیر اصلاح شده

# نتایج در پوشه‌ای به این نام در کنار اسکریپت ذخیره می‌شوند
OUTPUT_DIR = 'analysis_results_final_v2'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"نتایج در این پوشه ذخیره خواهند شد: {os.path.abspath(OUTPUT_DIR)}")

# --- 2. توابع کمکی ---

def load_and_combine_data(path, data_type):
    """تمام فایل‌های CSV یک نوع خاص (link یا mode) را از یک مسیر بارگذاری و ترکیب می‌کند."""
    pattern = os.path.join(path, f'*{data_type}*.csv')
    files = glob.glob(pattern)
    
    if not files:
        print(f"هشدار: هیچ فایلی با الگوی '{pattern}' در مسیر پیدا نشد.")
        return pd.DataFrame()
    
    print(f"فایل‌های پیدا شده برای نوع '{data_type}': {len(files)}")
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=['Time', 'Azimuth'])
            df_list.append(df)
        except Exception as e:
            print(f"خطا در خواندن فایل {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df.sort_values(by='Time').reset_index(drop=True)

def calculate_features(df):
    """ویژگی‌های تحلیل را محاسبه می‌کند."""
    if df.empty: return df
    df_out = df.copy()
    df_out['Time_diff'] = df_out['Time'].diff().fillna(0)
    df_out['Azimuth_diff'] = df_out['Azimuth'].diff().fillna(0)
    df_out['Velocity'] = (df_out['Azimuth_diff'] / df_out['Time_diff']).replace([np.inf, -np.inf], 0).fillna(0)
    df_out['Acceleration'] = (df_out['Velocity'].diff().fillna(0) / df_out['Time_diff']).replace([np.inf, -np.inf], 0).fillna(0)
    return df_out

def analyze_sudden_changes(df, report_file):
    """تغییرات ناگهانی زاویه در نقاط نزدیک به هم را تحلیل می‌کند."""
    report_file.write("\n--- تحلیل تغییرات ناگهانی زاویه (نقاط نزدیک) ---\n")
    time_diff_positive = df[df['Time_diff'] > 0]['Time_diff']
    if time_diff_positive.empty:
        report_file.write("امکان تحلیل وجود ندارد (هیچ فاصله زمانی مثبتی یافت نشد).\n")
        return
    threshold = time_diff_positive.quantile(0.25)
    nearby_df = df[(df['Time_diff'] > 0) & (df['Time_diff'] <= threshold)]
    if nearby_df.empty:
        report_file.write(f"هیچ نقطه‌ای با فاصله زمانی کمتر از آستانه ({threshold:.4f}s) یافت نشد.\n")
    else:
        report_file.write(f"آمار سرعت برای نقاط با فاصله زمانی کمتر از {threshold:.4f} ثانیه:\n")
        report_file.write(nearby_df['Velocity'].describe().to_string())
        report_file.write("\n\n")

def analyze_time_steps(df, report_file):
    """تحلیل جامع و دسته‌بندی شده گام‌های زمانی را انجام می‌دهد."""
    report_file.write("\n--- تحلیل جامع گام‌های زمانی (Time Steps) ---\n")
    if 'Time_diff' not in df.columns or df.empty:
        report_file.write("ستون Time_diff برای تحلیل یافت نشد.\n")
        return None
    time_diffs = df['Time_diff']
    report_file.write("آمار کامل فواصل زمانی (Time_diff):\n")
    report_file.write(time_diffs.describe().to_string())
    report_file.write("\n\n")
    bins = {
        'Zero or Negative (<_0s)': (time_diffs <= 0).sum(),
        'Very Fast (0-0.1s)': ((time_diffs > 0) & (time_diffs <= 0.1)).sum(),
        'Normal (0.1-1s)': ((time_diffs > 0.1) & (time_diffs <= 1.0)).sum(),
        'Short Gaps (1-5s)': ((time_diffs > 1.0) & (time_diffs <= 5.0)).sum(),
        'Large Gaps (>5s)': (time_diffs > 5.0).sum()
    }
    total_points = len(time_diffs)
    percentages = {key: (value / total_points) * 100 for key, value in bins.items()}
    report_file.write("دسته‌بندی رفتاری گام‌های زمانی و درصد فراوانی:\n")
    for category, pct in percentages.items():
        report_file.write(f"- {category:<25}: {pct:6.2f}%\n")
    report_file.write("\n")
    return percentages

def plot_time_step_distribution(gen_stats, real_stats, data_type, output_dir):
    """نمودار میله‌ای مقایسه‌ای برای توزیع دسته‌بندی شده گام‌های زمانی رسم می‌کند."""
    if gen_stats is None or real_stats is None: return
    labels, gen_values, real_values = list(gen_stats.keys()), list(gen_stats.values()), list(real_stats.values())
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, gen_values, width, label='Generated')
    rects2 = ax.bar(x + width/2, real_values, width, label='Real')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Time Step Behavior Distribution - {data_type.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{data_type}_timestep_distribution.png"))
    plt.close()

def plot_analysis(df_gen, df_real, gen_time_stats, real_time_stats, data_type):
    """تمام نمودارهای تحلیلی را برای یک نوع داده رسم و ذخیره می‌کند."""
    plot_path_prefix = os.path.join(OUTPUT_DIR, f"{data_type}_")
    plot_time_step_distribution(gen_time_stats, real_time_stats, data_type, OUTPUT_DIR)
    
    features = ['Time_diff', 'Azimuth', 'Velocity', 'Acceleration']
    fig, axes = plt.subplots(len(features), 2, figsize=(12, 16))
    fig.suptitle(f'Comparative Histograms for {data_type.upper()} Data', fontsize=16)
    for i, feature in enumerate(features):
        df_gen[feature].hist(ax=axes[i, 0], bins=100)
        axes[i, 0].set_title(f'Generated - {feature}'); axes[i, 0].set_yscale('log')
        df_real[feature].hist(ax=axes[i, 1], bins=100, color='orange')
        axes[i, 1].set_title(f'Real - {feature}'); axes[i, 1].set_yscale('log')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(plot_path_prefix + "histograms.png"); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})
    fig.suptitle(f'Azimuth Polar Distribution for {data_type.upper()} Data', fontsize=16)
    axes[0].hist(np.deg2rad(df_gen['Azimuth']), bins=180); axes[0].set_title('Generated')
    axes[1].hist(np.deg2rad(df_real['Azimuth']), bins=180, color='orange'); axes[1].set_title('Real')
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(plot_path_prefix + "polar_azimuth.png"); plt.close()

    plt.figure(figsize=(12, 6))
    try:
        kde_gen = gaussian_kde(df_gen['Time']); time_range_gen = np.linspace(df_gen['Time'].min(), df_gen['Time'].max(), 1000)
        plt.plot(time_range_gen, kde_gen(time_range_gen), label='Generated Density')
    except Exception as e: print(f"Could not plot KDE for Generated {data_type}: {e}")
    try:
        kde_real = gaussian_kde(df_real['Time']); time_range_real = np.linspace(df_real['Time'].min(), df_real['Time'].max(), 1000)
        plt.plot(time_range_real, kde_real(time_range_real), label='Real Density', color='orange', alpha=0.7)
    except Exception as e: print(f"Could not plot KDE for Real {data_type}: {e}")
    plt.title(f'Point Density Distribution (KDE) for {data_type.upper()} Data')
    plt.xlabel('Time'); plt.ylabel('Density'); plt.legend()
    plt.savefig(plot_path_prefix + "density_kde.png"); plt.close()

# --- 3. اجرای اصلی تحلیل ---

report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
with open(report_path, 'w', encoding='utf-8') as report:
    report.write("گزارش جامع تحلیل و مقایسه داده‌های واقعی و تولیدی (نسخه نهایی با خواندن تمام فایل‌ها)\n")
    report.write("="*80 + "\n\n")

    for data_type in ['link', 'mode']:
        report.write(f"شروع تحلیل برای نوع داده: {data_type.upper()}\n")
        report.write("-" * 50 + "\n\n")

        print(f"بارگذاری و تحلیل داده‌های {data_type.upper()}...")
        gen_df = load_and_combine_data(GENERATED_DATA_DIR, data_type)
        real_df = load_and_combine_data(REAL_DATA_DIR, data_type)

        if gen_df.empty or real_df.empty:
            print(f"ادامه تحلیل برای {data_type} ممکن نیست چون یکی از دیتافریم‌ها خوانده نشد.")
            report.write("یکی از مجموعه داده‌ها (واقعی یا تولیدی) خوانده نشد. تحلیل انجام نشد.\n\n")
            continue

        gen_featured = calculate_features(gen_df); real_featured = calculate_features(real_df)
        report.write("--- آمار توصیفی داده‌های تولیدی (Generated) ---\n"); report.write(gen_featured.describe().to_string()); report.write("\n\n")
        report.write("--- آمار توصیفی داده‌های واقعی (Real) ---\n"); report.write(real_featured.describe().to_string()); report.write("\n\n")
        
        gen_time_stats = analyze_time_steps(gen_featured, report); real_time_stats = analyze_time_steps(real_featured, report)
        
        report.write("--- تحلیل دامنه نوسانات آزیموت ---\n")
        report.write(f"انحراف معیار آزیموت (تولیدی): {gen_featured['Azimuth'].std():.4f}\n")
        report.write(f"انحراف معیار آزیموت (واقعی):   {real_featured['Azimuth'].std():.4f}\n")
        report.write(f"دامنه تغییرات آزیموت (تولیدی): {(gen_featured['Azimuth'].max() - gen_featured['Azimuth'].min()):.4f}\n")
        report.write(f"دامنه تغییرات آزیموت (واقعی):   {(real_featured['Azimuth'].max() - real_featured['Azimuth'].min()):.4f}\n\n")

        analyze_sudden_changes(gen_featured, report); analyze_sudden_changes(real_featured, report)

        print("تولید نمودارها...")
        plot_analysis(gen_featured, real_featured, gen_time_stats, real_time_stats, data_type)

        report.write(f"تحلیل برای {data_type.upper()} تکمیل شد.\n"); report.write("="*80 + "\n\n")
        print(f"تحلیل برای {data_type.upper()} تکمیل شد.")

print(f"\nتحلیل جامع با موفقیت انجام شد.")
print(f"گزارش متنی و نمودارها در پوشه زیر ذخیره شدند:\n{os.path.abspath(OUTPUT_DIR)}")