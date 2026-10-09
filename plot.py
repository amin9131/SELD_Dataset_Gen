import json
import matplotlib.pyplot as plt
import numpy as np

# تابعی برای حذف فاصله‌های اضافی از کلیدهای JSON (به دلیل وجود Space در انتهای کلیدهای فایل شما)
def clean_json_keys(obj):
    if isinstance(obj, dict):
        return {k.strip(): clean_json_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_keys(element) for element in obj]
    return obj

# 1. بارگذاری و خواندن فایل متادیتا
file_path = 'metadata.json'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    
data = clean_json_keys(raw_data)

# 2. تنظیمات نمودار سه‌بعدی
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 3. رسم موقعیت میکروفون‌ها (به عنوان مرجع محیطی)
mic_positions = data.get('mic_positions_m', {})
if mic_positions:
    mic_x = [pos[0] for pos in mic_positions.values()]
    mic_y = [pos[1] for pos in mic_positions.values()]
    mic_z = [pos[2] for pos in mic_positions.values()]
    ax.scatter(mic_x, mic_y, mic_z, c='black', marker='x', s=60, label='Microphones', zorder=5)

# 4. تابع کمکی برای رسم مسیرها
def plot_trajectories(sources, source_name, line_color):
    if not sources:
        return
    
    for idx, source in enumerate(sources):
        trajectory = source.get('trajectory_samples', [])
        if not trajectory:
            continue
        
        # استخراج مختصات X, Y, Z
        x = [sample['position_xyz_m'][0] for sample in trajectory]
        y = [sample['position_xyz_m'][1] for sample in trajectory]
        z = [sample['position_xyz_m'][2] for sample in trajectory]
        
        # رسم خط مسیر
        # (فقط برای اولین آیتم لیبل را تنظیم می‌کنیم تا در راهنما (Legend) تکرار نشود)
        lbl = source_name if idx == 0 else ""
        ax.plot(x, y, z, color=line_color, linewidth=2.5, label=lbl, alpha=0.8)
        
        # رسم نقطه شروع (دایره سبز)
        start_lbl = 'Start Point' if idx == 0 else ""
        ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=120, edgecolor='black', label=start_lbl, zorder=10)
        
        # رسم نقطه پایان (ستاره زرد)
        end_lbl = 'End Point' if idx == 0 else ""
        ax.scatter(x[-1], y[-1], z[-1], color='gold', marker='*', s=200, edgecolor='black', label=end_lbl, zorder=10)

# 5. رسم پهپادها و منابع مزاحم
plot_trajectories(data.get('drones', []), 'DJI Mini 2 (Drone)', 'blue')
plot_trajectories(data.get('known_unwanted', []), 'Known Unwanted', 'red')

# 6. تنظیمات ظاهری نمودار
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('3D Trajectories of Sound Sources', fontsize=14, fontweight='bold')

# مرتب‌سازی Legend برای جلوگیری از تکرار
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(0, 1))

# تنظیمات شبکه و پس‌زمینه
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 7. نمایش نمودار
plt.show()