import os
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    print("Авторизация на Hugging Face Hub...")
    login(token=hf_token)
else:
    print("⚠Внимание: HF_TOKEN не найден, могут быть ограничения rate limiting")

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = os.path.join(base_dir, 'docs', 'ImageNet', 'test')
os.makedirs(output_dir, exist_ok=True)

print("Загрузка датасета helenqu/ImageNet-Paste...")
ds = load_dataset(
    "helenqu/ImageNet-Paste",
    split="validation",
    token=hf_token
)

print(f"Датасет загружен. Всего записей: {len(ds)}")

for i, item in enumerate(ds):
    try:
        image = item['image']
        output_path = os.path.join(output_dir, f"image_{i}.jpg")
        image.save(output_path, 'JPEG')

        # Прогресс каждые 1000 изображений
        if (i + 1) % 1000 == 0:
            print(f"Сохранено {i + 1}/{len(ds)} изображений...")

    except Exception as e:
        print(f"Ошибка при сохранении изображения {i}: {e}")
        continue

print(f"\nГотово! Сохранено {len(ds)} изображений в {output_dir}")