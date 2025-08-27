import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# ---------- پروفایل گام‌های زمانی بر اساس تحلیل دادهٔ واقعی ----------
TIME_DIFF_PROFILE = {
    'link': {
        'probs': [0.1629, 0.4172, 0.3846, 0.0319, 0.0034],
        # بازه‌ها: صفر، بسیار سریع، عادی، فاصلهٔ کوتاه، فاصلهٔ بزرگ
        'ranges': [(0.0, 0.0), (0.0, 0.1), (0.1, 1.0), (1.0, 5.0), (5.0, 20.0)]
    },
    'mode': {
        'probs': [0.6011, 0.0366, 0.1648, 0.1656, 0.0319],
        'ranges': [(0.0, 0.0), (0.0, 0.1), (0.1, 1.0), (1.0, 5.0), (5.0, 20.0)]
    }
}


# ---------- پارامترهای ورودی ----------
num_files = int(input('Enter number of files to generate: '))
rows_per_file = int(input('Enter approximate number of rows per file: '))
dataset_type = input('Enter dataset type (link or mode): ').lower()

# خروجی به‌صورت پیش‌فرض در پوشه‌ای محلی ذخیره می‌شود تا وابسته به مسیر خاصی نباشد
output_dir = os.path.join(os.getcwd(), 'generated_data')
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ---------- توابع تولید دیتا ----------
def sample_time_steps(num_points, profile):
    """Generate time steps using a mixture distribution matching real data."""
    probs = profile['probs']
    ranges = profile['ranges']
    cats = np.random.choice(len(probs), size=num_points - 1, p=probs)
    steps = np.empty(num_points - 1)
    for i, c in enumerate(cats):
        low, high = ranges[c]
        if low == high == 0.0:
            steps[i] = 0.0
        else:
            steps[i] = np.random.uniform(low, high)
    return steps


def generate_object_points(num_points, behavior, mean_step, start_time, mean_az, sd_az, min_az, max_az, label, time_profile):
    steps = sample_time_steps(num_points, time_profile)
    t = np.cumsum([start_time] + list(steps))
    offset = np.clip(np.random.normal(mean_az, sd_az), min_az, max_az)

    if behavior == 'sinusoidal':
        A = random.uniform(5, 15)
        f = random.uniform(0.001, 0.01)
        phi = random.uniform(0, 2 * np.pi)
        azimuth = offset + A * np.sin(2 * np.pi * f * t + phi)

    elif behavior == 'linear':
        v = random.uniform(-0.01, 0.01)
        azimuth = offset + v * (t - start_time)

    elif behavior == 'mixed':
        split = random.uniform(0.3, 0.7)
        num1 = int(num_points * split)
        t1, t2 = t[:num1], t[num1:]
        A = random.uniform(5, 15)
        f = random.uniform(0.001, 0.01)
        phi = random.uniform(0, 2 * np.pi)
        v = random.uniform(-0.005, 0.005)
        az1 = offset + A * np.sin(2 * np.pi * f * t1 + phi) if num1 > 0 else np.array([])
        b = az1[-1] - v * (t1[-1] - start_time) if num1 > 0 else offset
        az2 = b + v * (t2 - start_time)
        azimuth = np.concatenate([az1, az2])

    azimuth = (azimuth % 360 + 360) % 360
    return pd.DataFrame({'Time': t, 'Azimuth': azimuth, 'Label': label})

def apply_density_variation(df):
    """کاهش یا افزایش موضعی چگالی نقاط.

    برای جلوگیری از تولید بیش از حد گام‌های زمانی بسیار کوچک، تنها عملیات کاهش تراکم
    انجام می‌شود تا توزیع فاصله‌های زمانی به داده‌های واقعی نزدیک‌تر بماند."""
    total_time = df['Time'].max() - df['Time'].min()
    num_windows = random.randint(2, 5)

    for _ in range(num_windows):
        win_size = random.uniform(0.05, 0.2) * total_time
        start_t = random.uniform(df['Time'].min(), df['Time'].max() - win_size)
        win_mask = (df['Time'] >= start_t) & (df['Time'] <= start_t + win_size)

        if win_mask.sum() < 5:
            continue

        # تنها کاهش تراکم در پنجره انتخابی اعمال می‌شود
        reduction = random.uniform(0.1, 0.8)
        mask = np.random.choice([True, False], win_mask.sum(), p=[1 - reduction, reduction])
        df = df.drop(df[win_mask].index[~mask])

        df = df.sort_values('Time').reset_index(drop=True)
    return df.reset_index(drop=True)

def apply_gaps(df):
    total_time = df['Time'].max() - df['Time'].min()
    num_gaps = random.randint(1, 3)
    for _ in range(num_gaps):
        gap_size = random.uniform(0.1, 0.4) * total_time
        start_time = random.uniform(df['Time'].min(), df['Time'].max() - gap_size)
        df = df[(df['Time'] < start_time) | (df['Time'] > start_time + gap_size)]
    return df.reset_index(drop=True)

def apply_distortion(df):
    df['Time'] += np.random.normal(0, 0.01, len(df))
    df['Azimuth'] += np.random.normal(0, 1, len(df))
    return df

# ------------------ بخش‌های افزوده‌شده ------------------
# ثابت‌های تجربی از داده‌ی واقعی (تحلیل قبلی)
REAL_CONF = {
    'link': {
        'median_dt': 0.123,                  # میانه Δt واقعی (واحد خام)
        'max_per_median_window': 19,         # بیشینه تراکم در پنجره با عرض میانه Δt
        'pct_points_in_exact_same_time': 0.1629,  # درصد نقاط با زمان تکراری در داده واقعی
        'max_exact_same_time': 19            # بیشینه نقاط دقیقاً در یک مقدار Time
    },
    'mode': {
        'median_dt': 0.88725,
        'max_per_median_window': 10,
        'pct_points_in_exact_same_time': 0.6011,  # درصد نقاط با زمان تکراری در داده واقعی
        'max_exact_same_time': 8
    }
}

def clamp_density_median_window(df, window_width, max_per_bin):
    """
    در هر پنجره با عرض window_width، اگر تعداد نقاط از max_per_bin بیشتر بود
    به‌صورت تصادفی نقاط اضافه حذف می‌شوند تا از سقف عبور نکند.
    """
    if window_width <= 0 or not np.isfinite(window_width) or df.empty:
        return df
    t0 = df['Time'].min()
    bins = np.floor((df['Time'].values - t0) / window_width).astype(np.int64)
    df = df.copy()
    df['__bin'] = bins
    vc = df['__bin'].value_counts()
    over_bins = vc[vc > max_per_bin].index.tolist()
    for b in over_bins:
        idx = df.index[df['__bin'] == b].to_numpy()
        drop_n = int(len(idx) - max_per_bin)
        if drop_n > 0:
            drop_idx = np.random.choice(idx, size=drop_n, replace=False)
            df = df.drop(drop_idx)
    return df.drop(columns='__bin').sort_values('Time').reset_index(drop=True)

def inject_exact_same_time_percent(df, target_fraction, window_width, max_per_bin, max_exact_same_time, max_neighbors_shift=10):
    """
    درصدی از نقاط را (target_fraction) وارد گروه‌های هم‌زمان می‌کند.
    برای هر «نقطه لنگر»، تا حداکثر 10 همسایه (تصادفی) به همان مقدار Time منتقل می‌شوند.
    رعایت می‌شود که:
      - تعداد نقاط در پنجره‌ی عرض=window_width از max_per_bin بیشتر نشود
      - تعداد نقاط دقیقا در یک مقدار Time از max_exact_same_time عبور نکند
    """
    if df.empty or target_fraction <= 0:
        return df
    df = df.sort_values('Time').reset_index(drop=True).copy()
    N = len(df)
    target_multi_points = int(round(target_fraction * N))

    # وضعیت فعلیِ چند-زمانی‌های دقیق
    time_counts = df['Time'].value_counts()
    current_multi_points = int(time_counts[time_counts >= 2].sum())
    remaining = max(0, target_multi_points - current_multi_points)
    if remaining <= 0:
        return df

    t = df['Time'].values
    t0 = t.min()
    bins = np.floor((t - t0) / window_width).astype(np.int64)
    bin_counts = pd.Series(bins).value_counts().to_dict()
    time_counts = pd.Series(t).value_counts().to_dict()

    # ترتیب تصادفی برای انتخاب لنگرها
    order = np.arange(N)
    np.random.shuffle(order)

    for anchor_idx in order:
        if remaining <= 0:
            break
        anchor_time = float(t[anchor_idx])
        anchor_bin = int(bins[anchor_idx])

        # ظرفیت‌های مجاز
        room_exact = max_exact_same_time - int(time_counts.get(anchor_time, 0))
        room_bin   = max_per_bin - int(bin_counts.get(anchor_bin, 0))
        cap_here   = min(room_exact, room_bin, max_neighbors_shift, remaining)
        if cap_here <= 0:
            continue

        # همسایه‌ها را از چپ و راست با یک پنجرهٔ نمایه‌ای محدود انتخاب می‌کنیم
        neighbor_span = 20
        left = list(range(max(0, anchor_idx - neighbor_span), anchor_idx))
        right = list(range(anchor_idx + 1, min(N, anchor_idx + neighbor_span + 1)))
        candidates = [i for i in (left[::-1] + right) if t[i] != anchor_time]
        if not candidates:
            continue

        num_to_move = np.random.randint(1, min(10, cap_here) + 1)
        chosen = np.random.choice(candidates, size=num_to_move, replace=False)

        # به مقدار زمانِ لنگر منتقل می‌کنیم (آزیموت دست‌نخورده)
        df.loc[chosen, 'Time'] = anchor_time

        # شمارنده‌ها را به‌روزرسانی می‌کنیم
        for i in chosen:
            old_bin = int(bins[i])
            bin_counts[old_bin] = bin_counts.get(old_bin, 1) - 1
            bins[i] = anchor_bin  # به سبد لنگر منتقل شد
        bin_counts[anchor_bin] = bin_counts.get(anchor_bin, 0) + num_to_move
        time_counts[anchor_time] = time_counts.get(anchor_time, 1) + num_to_move
        remaining -= num_to_move

    return df.sort_values('Time').reset_index(drop=True)

def balance_time_step_distribution(df, profile, t_max):
    """Adjust time differences so category frequencies stay close to target profile.

    Categories that exceed their expected counts are shifted to the next
    category range by increasing subsequent time stamps with small random
    increments. This preserves randomness while preventing domination of
    very small time steps.
    """
    if len(df) < 2:
        return df

    ranges = profile['ranges']
    probs = profile['probs']
    times = df['Time'].to_numpy(copy=True)
    diffs = np.diff(times)
    # map each diff to a category index
    def cat_idx(x):
        for idx, (low, high) in enumerate(ranges):
            if low <= x <= high:
                return idx
        return len(ranges) - 1

    cats = np.array([cat_idx(d) for d in diffs])
    total = len(diffs)
    desired = (np.array(probs) * total).astype(int)
    counts = np.bincount(cats, minlength=len(probs))

    for idx in range(len(probs) - 1):
        excess = counts[idx] - desired[idx]
        if excess <= 0:
            continue
        locs = np.where(cats == idx)[0]
        if len(locs) == 0:
            continue
        move = np.random.choice(locs, size=excess, replace=False)
        for i in move:
            new_step = np.random.uniform(ranges[idx + 1][0], ranges[idx + 1][1])
            delta = new_step - diffs[i]
            times[i + 1 :] += delta
            diffs[i] = new_step
            cats[i] = idx + 1
        counts[idx] -= excess
        counts[idx + 1] += excess

    df['Time'] = times
    df = df[df['Time'] <= t_max].reset_index(drop=True)
    return df

# ---------- پارامترها بر اساس نوع دیتاست ----------
if dataset_type == 'link':
    start_time = 58154.7921830403
    end_time   = 67414.73985 
    mean_step = 0.3996
    mean_az = 238.276
    sd_az = (281.261 - 123.243) / 6
    min_az = 123.243
    max_az = 281.261
elif dataset_type == 'mode':
    start_time = 58156.48918304
    end_time   = 67414.96685
    mean_step = 0.8576
    mean_az = 228.138
    sd_az = (329.909 - 105.405) / 12
    min_az = 105.405
    max_az = 329.909
else:
    raise ValueError("Invalid dataset type. Use 'link' or 'mode'.")

CONF = REAL_CONF[dataset_type]

# ---------- تولید و پلات برای هر فایل ----------
for file_idx in range(num_files):
    combined_df = pd.DataFrame()
    num_objects = random.randint(2, 10)

    num_sin = max(1, num_objects // 2)
    behaviors = ['sinusoidal'] * num_sin
    for _ in range(num_objects - num_sin):
        behaviors.append(random.choice(['linear', 'mixed']))
    random.shuffle(behaviors)

    for obj_id, behavior in enumerate(behaviors):
        points_per_object = rows_per_file // num_objects
        df_obj = generate_object_points(points_per_object, behavior, mean_step, start_time,
                                        mean_az, sd_az, min_az, max_az, obj_id,
                                        TIME_DIFF_PROFILE[dataset_type])

        df_obj = apply_density_variation(df_obj)
        if random.random() < 0.5:
            df_obj = apply_gaps(df_obj)
        df_obj = apply_distortion(df_obj)

        combined_df = pd.concat([combined_df, df_obj])

    combined_df = combined_df.sort_values(by='Time').reset_index(drop=True)

    # --- محدود کردن بازهٔ زمانی ---
    combined_df = combined_df[
        (combined_df['Time'] >= start_time) &
        (combined_df['Time'] <= end_time)
    ].reset_index(drop=True)

    # === (1) جلوگیری از عبور چگالی از بیشینه‌ی دیتاست واقعی (پنجره=میانه Δt واقعی) ===
    combined_df = clamp_density_median_window(
        combined_df, 
        window_width=CONF['median_dt'], 
        max_per_bin=CONF['max_per_median_window']
    )

    # === (2) تزریق هم‌زمانیِ دقیق با همان درصد واقعی، با رعایت سقف‌های چگالی ===
    combined_df = inject_exact_same_time_percent(
        combined_df,
        target_fraction=CONF['pct_points_in_exact_same_time'],
        window_width=CONF['median_dt'],
        max_per_bin=CONF['max_per_median_window'],
        max_exact_same_time=CONF['max_exact_same_time'],
        max_neighbors_shift=10
    )

    # === (3) تعدیل نهایی توزیع گام‌های زمانی برای نزدیک‌تر شدن به واقعیت ===
    combined_df = balance_time_step_distribution(
        combined_df,
        TIME_DIFF_PROFILE[dataset_type],
        t_max=end_time
    )

    # ذخیره CSV
    output_path = os.path.join(output_dir, f'generated_{dataset_type}_{file_idx}.csv')
    combined_df.to_csv(output_path, index=False)
    print(f'Generated {output_path}')

    # پلات مخصوص همین فایل
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.scatter(combined_df['Time'], combined_df['Azimuth'], s=1,
                c='b' if dataset_type == 'link' else 'orange',
                label=f'{dataset_type.capitalize()} Data')
    plt.xlabel('Time')
    plt.ylabel('Azimuth (degrees)')
    plt.title('Azimuth vs Time')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(combined_df['Azimuth'], bins=50, alpha=0.7,
             label=f'{dataset_type.capitalize()} Data')
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Frequency')
    plt.title('Azimuth Distribution')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'generated_{dataset_type}_{file_idx}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")

print('All datasets generated and plotted separately.')
