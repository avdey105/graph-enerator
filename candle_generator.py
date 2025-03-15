import sys
import subprocess
import argparse
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image
import gc  # Добавлен импорт сборщика мусора

# Проверка и установка зависимостей
required_packages = [
    ('pandas', 'pandas'),
    ('mplfinance', 'mplfinance'),
    ('matplotlib', 'matplotlib'),
    ('tqdm', 'tqdm'),
    ('Pillow', 'Pillow')
]

missing_packages = []
for (pkg, import_name) in required_packages:
    try:
        __import__(import_name)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print("Установка недостающих пакетов:", ', '.join(missing_packages))
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Генератор свечных графиков из CSV данных')
parser.add_argument('input_file', type=str, help='Путь к CSV файлу с данными')
parser.add_argument('-o', '--output', type=str, default='output_images', 
                    help='Папка для сохранения изображений (по умолчанию: output_images)')
parser.add_argument('-w', '--window', type=int, default=200,
                    help='Размер окна данных (по умолчанию: 200)')
parser.add_argument('-s', '--step', type=int, default=10,
                    help='Шаг смещения окна (по умолчанию: 10)')
parser.add_argument('-d', '--dpi', type=int, default=160,
                    help='Качество изображений (DPI) (по умолчанию: 80)')
parser.add_argument('-b', '--batch', type=int, default=200,
                    help='Размер пакета для обработки (по умолчанию: 200)')

args = parser.parse_args()

# Загрузка данных
try:
    df = pd.read_csv(
        args.input_file,
        parse_dates=['Open time'],
        index_col='Open time'
    ).sort_index()
except Exception as e:
    print(f"Ошибка загрузки файла: {str(e)}")
    sys.exit(1)

# Настройка стиля графиков
mc = mpf.make_marketcolors(
    up='#26A69A',
    down='#EF5350',
    wick={'up': '#26A69A', 'down': '#EF5350'},
    volume='in'
)

style = mpf.make_mpf_style(
    marketcolors=mc,
    gridstyle='',
    facecolor='white',
    figcolor='white',
    y_on_right=False,
    rc={
        'axes.spines.bottom': False,
        'axes.spines.left': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': 'white',
        'ytick.color': 'white',
        'lines.linewidth': 0.8
    }
)

# Создание папки для результатов
os.makedirs(args.output, exist_ok=True)

# Параметры генерации
WINDOW_SIZE = args.window
STEP = args.step
BATCH_SIZE = args.batch
total_bars = len(df)
start_indices = list(range(0, total_bars - WINDOW_SIZE + 1, STEP))

def generate_plot(start_idx):
    try:
        end_idx = start_idx + WINDOW_SIZE
        window_data = df.iloc[start_idx:end_idx]
        last_time = window_data.index[-1]
        
        filename = f"btc_{last_time.strftime('%Y%m%d_%H%M')}.png"
        file_path = os.path.join(args.output, filename)
        
        if os.path.exists(file_path):
            return
        
        fig, _ = mpf.plot(
            window_data,
            type='candle',
            style=style,
            volume=True,
            figratio=(12, 6),
            returnfig=True,
            show_nontrading=True,
            axisoff=True,
            savefig=dict(
                fname=file_path,
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=args.dpi
            )
        )
        plt.close(fig)
        plt.close('all')  # Дополнительная очистка
        
        # Обрезка изображения
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Параметры обрезки
                left = 78
                right = width - 78
                upper = 0
                lower = height - 20
                
                if right > left and lower > upper:
                    cropped_img = img.crop((left, upper, right, lower))
                    cropped_img.save(file_path, quality=100)
        except Exception as e:
            print(f"Ошибка при обрезке {filename}: {str(e)}")

    except Exception as e:
        print(f"Ошибка при обработке индекса {start_idx}: {str(e)}")
    finally:
        gc.collect()  # Принудительная очистка памяти

if __name__ == '__main__':
    print(f"Начало генерации изображений...")
    print(f"Всего будет создано изображений: {len(start_indices)}")
    
    # Ограничение количества процессов
    num_workers = max(2, min(cpu_count(), 4))  # Не более 4 процессов
    print(f"Используется ядер CPU: {num_workers}")
    
    processed_files = set(os.listdir(args.output))
    filtered_indices = []
    
    for idx in start_indices:
        end_idx = idx + WINDOW_SIZE
        last_time = df.index[end_idx-1]
        filename = f"btc_{last_time.strftime('%Y%m%d_%H%M')}.png"
        if filename not in processed_files:
            filtered_indices.append(idx)
    
    print(f"Осталось обработать: {len(filtered_indices)} изображений")
    
    # Разбивка на пакеты
    for i in range(0, len(filtered_indices), BATCH_SIZE):
        batch = filtered_indices[i:i+BATCH_SIZE]
        print(f"Обработка пакета {i//BATCH_SIZE + 1}/{(len(filtered_indices)-1)//BATCH_SIZE + 1}")
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_plot, batch, chunksize=10),
                total=len(batch),
                desc="Прогресс",
                unit="image"
            ))
        
        # Очистка памяти между пакетами
        gc.collect()
    
    print(f"Генерация завершена! Изображения сохранены в папку: {args.output}")
