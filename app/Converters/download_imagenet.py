import os
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from collections import defaultdict

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    print("Авторизация на Hugging Face Hub...")
    login(token=hf_token)
else:
    print("Внимание: HF_TOKEN не найден, могут быть ограничения rate limiting")

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

print(f"Ключи первой записи: {ds[0].keys()}")
if 'label' in ds[0]:
    print(f"Пример метки: {ds[0]['label']}")
    print(f"Тип метки: {type(ds[0]['label'])}")

print("\nСоздание папок для классов 00000-00999 в test/...")
for i in range(1000):
    class_folder = os.path.join(output_dir, f"{i:05d}")
    os.makedirs(class_folder, exist_ok=True)
print("Папки созданы!")

counters = defaultdict(int)

total_images = len(ds)
images_per_class = 50
max_images = 1000 * images_per_class

print(f"\nНачинаем сохранение {min(total_images, max_images)} изображений...")
print(f"По {images_per_class} изображений в каждой из 1000 папок")
print(f"Путь: {output_dir}/00000/ ... /00999/")

saved_count = 0
skipped_count = 0

for i, item in enumerate(ds):
    if saved_count >= max_images:
        print(f"\nДостигнут лимит в {max_images} изображений")
        break

    try:
        image = item['image']
        if 'label' in item:
            class_id = item['label']
        elif 'labels' in item:
            class_id = item['labels']
        else:
            skipped_count += 1
            if skipped_count % 100 == 0:
                print(f"Пропущено {skipped_count} изображений: нет метки класса")
            continue

        if not isinstance(class_id, (int, float)) or class_id < 0 or class_id >= 1000:
            skipped_count += 1
            if skipped_count % 100 == 0:
                print(f"Пропущено {skipped_count} изображений: некорректный class_id {class_id}")
            continue

        class_id_int = int(class_id)

        if counters[class_id_int] >= images_per_class:
            continue

        class_folder = os.path.join(output_dir, f"{class_id_int:05d}")
        filename = f"image_{counters[class_id_int]}.jpg"
        output_path = os.path.join(class_folder, filename)

        image.save(output_path, 'JPEG')
        counters[class_id_int] += 1
        saved_count += 1

    except Exception as e:
        print(f"Ошибка при сохранении изображения {i}: {e}")
        continue

print(f"\n{'=' * 50}")
print(f"ГОТОВО!")
print(f"{'=' * 50}")
print(f"Всего сохранено: {saved_count} изображений")
print(f"Пропущено: {skipped_count} изображений")
print(f"Распределение по первым 10 классам:")
for class_id in range(10):
    print(f"  Класс {class_id:05d}: {counters[class_id]} изображений")

classes_with_50 = sum(1 for c in range(1000) if counters[c] == 50)
print(f"\nКлассов с ровно 50 изображениями: {classes_with_50}/1000")

if classes_with_50 < 1000:
    print("\nКлассы с недостаточным количеством:")
    for class_id in range(1000):
        if counters[class_id] < 50 and counters[class_id] > 0:
            print(f"  Класс {class_id:05d}: только {counters[class_id]} изображений")

print(f"\nПуть к данным: {output_dir}")
print(f"Пример: {output_dir}/00042/image_0.jpg")