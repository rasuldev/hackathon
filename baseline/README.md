# Бейзлайн для задачи ранжирования

Для запуска предлагается использовать менеджер пакетов `poetry`. Зависимости описаны в корневой директории репозитория – `pyproject.toml`.

1. Скачать файлы [отсюда](https://disk.yandex.ru/d/au7AVG_3nTyUIw) и поместить в папку `./data`.

2. Запуск обучения (нужно запускать из `./baseline`):
    ```bash
    poetry run python train.py
    ```

Скрипт подготовит данные и фичи, обучит `CatBoostRanker` и выведет значение целевой метрики – [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain).

Куда можно двигаться, чтобы улучшить результат:
- посчитать факторы-счетчики такие, как:
  - статистики по донатам пользователя (в какие категории/фонды/дни недели предпочитает донатить)
  - статистики по фондам
  - по кейвордам и так далее
- использовать текстовые факторы (тайтлы и описания сборов), как с помощью стандартных средств `CatBoost`, так и с помощью любых других (`huggingface`, `nltk` и так далее)
- придумайте как использовать комментарии и лайки пользователей (`comments.csv` и `commentlikes.csv`)
- фичи, связанные со временем

---

**Важно**: какой бы алгоритм вы не использовали, финальный замер NDCG должен проводиться на третьем датафрейме (test_df), который нужно получить так:
```python
from dataset import prepare_raw_dataset, split_data_by_shares 

sessions = prepare_raw_dataset()
_, _, test_df = split_data_by_shares(
    sessions, {"train": 0.7, "val": 0.15, "test": 0.15}
)
```

Если вы не используете `CatBoost`, реализацию `ndcg` можно взять из [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).
