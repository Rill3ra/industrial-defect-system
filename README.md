# Industrial Defect System

Система обнаружения дефектов промышленных изделий на основе компьютерного зрения и глубокого обучения.

Поддерживает **MVTec AD** (категории `metal_nut` и `screw`) и несколько моделей классификации + PatchCore для детекции аномалий.

---

## Демо

<!--
Как вставить сюда свой GIF:
1. Создайте Issue в репозитории на GitHub (закрывать/публиковать не обязательно).
2. Перетащите GIF-файл в текстовое поле — GitHub сам загрузит его и подставит
   готовую ссылку вида ![demo](https://github.com/user-attachments/assets/...).
3. Скопируйте эту строку сюда вместо плейсхолдера ниже.
-->

![Demo]<img width="800" height="450" alt="video" src="https://github.com/user-attachments/assets/35780822-bd0b-4a57-ad97-3076d1636bfd" />


---

## Возможности

- **FastAPI** — REST API для инференса и генерации PDF-отчётов
- **Streamlit** — веб-интерфейс с загрузкой изображения, heatmap-визуализацией аномалий и bbox-локализацией дефекта
- Поддержка моделей: ResNet50, EfficientNet-B0, MobileNet-V3, ViT-B/16, DenseNet121, **PatchCore**
- Обучение, оценка и сравнение моделей
- Визуализации: heatmaps аномалий, ROC-кривые, confusion matrix
- Автоматическая генерация PDF-отчётов по результатам инференса

---

## Требования

- Docker и Docker Compose (v2, команда `docker compose`)
- ~5 ГБ свободного места (образы + веса моделей + кэш torch)
- Для локальной разработки без Docker: Python 3.11+

---

## Быстрый запуск (Docker)

### 1. Клонируйте репозиторий

```bash
git clone <URL_вашего_репозитория>
cd industrial-defect-system
```

### 2. Убедитесь, что веса моделей на месте

Docker-контейнеры монтируют `runs/` и `data/` с хоста как volumes (веса и датасет в образ не копируются). Перед запуском проверьте, что в проекте есть:

```
runs/checkpoints/
├── densenet121.pth
├── efficientnet_b0.pth
├── mobilenet_v3.pth
├── patchcore_memory_bank.pt
├── resnet50.pth
└── vit_b16.pth
```

### 3. Запуск через Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Сервис `streamlit` стартует только после того, как `api` пройдёт healthcheck (`/health`), поэтому при первом запуске подождите 10–30 секунд.

### 4. Откройте в браузере

- **API документация (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit интерфейс**: [http://localhost:8501](http://localhost:8501)

### 5. Проверка и остановка

```bash
# Логи
docker logs -f industrial-api
docker logs -f industrial-streamlit

# Статус контейнеров
docker compose -f docker/docker-compose.yml ps

# Остановка
docker compose -f docker/docker-compose.yml down

# Пересборка после изменений в коде
docker compose -f docker/docker-compose.yml up --build -d
```

> Оба контейнера используют общий кэш `~/.cache/torch` с хоста — это ускоряет повторные сборки и запуски, так как torch не перезагружает предобученные веса каждый раз.

---

## Локальная разработка (без Docker)

```bash
# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск API
export PYTHONPATH=$(pwd)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Запуск Streamlit (в отдельном терминале)
export API_URL=http://localhost:8000
streamlit run src/demo/app.py
```

---

## Структура проекта

```
industrial-defect-system/
├── docker/                     # Dockerfile.api, Dockerfile.streamlit, docker-compose.yml
├── src/
│   ├── api/                    # FastAPI приложение (app.py)
│   ├── demo/                   # Streamlit интерфейс (app.py)
│   ├── models/                 # PatchCore, postprocessing, model_factory
│   ├── inference/               # pipeline.py, predict.py, схемы
│   ├── data/                    # датасеты, трансформации, сплиты
│   ├── training/                 # скрипты обучения моделей
│   ├── evaluation/               # оценка и сравнение моделей
│   ├── visualization/             # bbox, heatmaps, ROC, confusion matrix
│   ├── report/                    # генерация PDF-отчётов
│   └── utils/                     # вспомогательные функции (heatmap.py)
├── data/
│   ├── raw/mvtec_ad/               # metal_nut, screw (train/test/ground_truth)
│   └── splits/                     # train.json, val.json, test.json
├── runs/
│   ├── checkpoints/                # веса моделей (*.pth, *.pt)
│   ├── logs/, metrics/, plots/      # результаты обучения и оценки
│   └── final_model_comparison.csv
├── configs/                          # конфигурационные файлы
├── requirements.txt
└── docker-compose.yml (см. docker/)
```

---

## Поддерживаемые модели

| Тип | Модель |
|---|---|
| Классификаторы | ResNet50, EfficientNet-B0, MobileNet-V3, ViT-B/16, DenseNet121 |
| Anomaly Detection | PatchCore |

Финальное решение (`decision`) комбинирует anomaly score от PatchCore и вероятность от классификатора — см. `src/inference/pipeline.py`.

---

## API

### `GET /health`
Проверка доступности сервиса.

### `POST /predict`
Принимает изображение (`multipart/form-data`, поле `file`), возвращает:

```json
{
  "label": 1,
  "decision": "defect",
  "classifier_prob": 0.87,
  "patchcore_score": 0.12,
  "bbox": {"x1": 2, "y1": 1, "x2": 6, "y2": 5, "score": 0.91},
  "heatmap": [[...]]
}
```

Полная документация и возможность протестировать эндпоинт — на `/docs`.

---

## Данные и веса моделей

Датасет MVTec AD и веса обученных моделей **не хранятся в репозитории** (см. `.gitignore`) — они слишком большие для git и распространяются отдельно.

### Датасет MVTec AD

1. Скачайте нужные категории (`metal_nut`, `screw`) с официального сайта: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. Распакуйте в:
   ```
   data/raw/mvtec_ad/metal_nut/
   data/raw/mvtec_ad/screw/
   ```
3. Датасет распространяется **только для некоммерческих исследовательских целей** — см. `license.txt` внутри каждой категории.

### Веса моделей

Веса (`runs/checkpoints/*.pth`, `patchcore_memory_bank.pt`) нужно либо:

- **обучить самостоятельно**:
  ```bash
  bash train_all.sh
  ```
  (потребуется скачанный датасет, см. выше)

- **либо получить готовые файлы отдельно** (например, через облачное хранилище/файлообменник) и положить в:
  ```
  runs/checkpoints/
  ├── densenet121.pth
  ├── efficientnet_b0.pth
  ├── mobilenet_v3.pth
  ├── patchcore_memory_bank.pt
  ├── resnet50.pth
  └── vit_b16.pth
  ```

Без этих файлов API и Streamlit-интерфейс не смогут выполнять инференс.

---

## Публикация в Git

```bash
cd industrial-defect-system
git init                      # если репозиторий ещё не инициализирован
git remote add origin <URL_вашего_репозитория>
git add .
git commit -m "Initial commit: industrial defect detection system"
git branch -M main
git push -u origin main
```

`.gitignore` уже настроен так, чтобы не коммитить: виртуальное окружение, датасет MVTec AD, чекпоинты моделей, логи и метрики (`runs/`) — они слишком большие для репозитория. Веса моделей и датасет нужно распространять отдельно (например, через Git LFS, облачное хранилище или скрипт загрузки).

---

## Авторы

Rill3ra
Разработано в рамках практики 2026.
