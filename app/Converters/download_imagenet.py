# app/Converters/download_imagenet.py
import os
from datasets import load_dataset
from PIL import Image


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = os.path.join(base_dir, 'docs', 'imagenet-paste', 'validation')
os.makedirs(output_dir, exist_ok=True)

print("Загрузка датасета helenqu/ImageNet-Paste...")
ds = load_dataset(
    "helenqu/ImageNet-Paste",
    split="validation",
    trust_remote_code=True
)

print(f"Датасет загружен. Всего записей: {len(ds)}")

for i, item in enumerate(ds):
    try:
        image = item['image']

        output_path = os.path.join(output_dir, f"image_{i}.jpg")
        image.save(output_path, 'JPEG')

    except Exception as e:
        print(f"Ошибка при сохранении изображения {i}: {e}")
        continue

print(f"\n✅ Готово! Сохранено {len(ds)} изображений в {output_dir}")
print(f"Размер датасета: {len(ds)} изображений")