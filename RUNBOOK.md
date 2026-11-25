# ПолярТех: запуск готового проекта

В переданном архиве завершены шесть модулей в `src/assistant/graphrag/` и `src/assistant/agent/`.
Исходные файлы инфраструктуры и контрактные тесты сохранены. На Python 3.12.16 прошли все 142 теста и `ruff check .`.

## Что нужно на вашем компьютере

- Docker + Docker Compose (на Windows запуск `make` удобнее выполнять в WSL).
- Ollama с моделями `qwen3:4b-instruct-2507-q4_K_M` и `bge-m3`.
- `make` и достаточное место на диске.

## Запуск и результаты

Скопируйте `.env.example` в `.env`, настройте `LLM_BASE_URL` по инструкции `docs/setup.md`. Если Ollama установлена на компьютере, выполните:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull bge-m3
```

В корневом каталоге проекта:

```bash
make up
make test
make lint
make index
make ask MODE=agent
make eval-all
make eval-extraction
```

### Windows без Docker и WSL

Установите Python 3.12, `uv` и Ollama. Загрузите обе модели командами выше. В первом окне PowerShell запустите ERP из корня проекта:

```powershell
uv sync --frozen --python 3.12
uv run python -m uvicorn erp.app:app --host 127.0.0.1 --port 8100
```

Во втором окне из того же каталога выполните:

```powershell
$env:LLM_BASE_URL = 'http://127.0.0.1:11434/v1'
$env:ERP_URL = 'http://127.0.0.1:8100'
uv run python -m assistant index
uv run python -m assistant ask --mode agent 'Ошибка E-17 на AR-210. Что заказать и сколько стоит в Казани?'
foreach ($mode in 'baseline', 'local', 'global', 'agent') {
    uv run python -m assistant eval --mode $mode
    if ($LASTEXITCODE -ne 0) { throw "Не завершена оценка $mode" }
}
uv run python -m assistant eval-extraction
```

### Результаты

В `artifacts/` должны появиться:

- `graph.json`, `communities.json`, `index_stats.json`;
- `eval_baseline.json`, `eval_local.json`, `eval_global.json`, `eval_agent.json`;
- `eval_extraction.json`;
- `llm_cache/`.

Четыре `eval_*.json` содержат метрики по 24 вопросам, в том числе точность по типам, Recall источников, долю выдуманных фактов, число вызовов и токенов. Перенесите результаты в `REPORT.pdf` перед формальной сдачей. Данный файл — промежуточный отчёт: в нём честно указано, что прогон с Ollama ещё не выполнен.

## Заказ

По умолчанию запросы работают без подтверждения (`confirm=False`). Агент возвращает `needs_confirmation=True` и `pending_action`. **Заказ в ERP не создаётся.**

Чтобы подтвердить заказ, пользователь должен отдельно согласовать действие и количество, затем вызывающая программа передаёт `confirm=True`.

## Проверки, выполненные при подготовке

- `PYTHONPATH=src python -m pytest -q`: **142 passed**.
- `tests/contract/test_tools_and_agent.py`: **24 passed**.
- `ruff check .`: ошибок нет.
- Учебная ERP запущена локально и вернула четыре склада и цену 2630 руб. для `P-1042B`.
- Метрики четырёх режимов и качества извлечения в приложенном черновике ещё отсутствуют.

## Для сдачи

В соответствии с `docs/submission.md`, нужно добавить публичную ссылку на GitHub, заполнить фактические результаты в `REPORT.pdf`, сохранить результаты в `artifacts/`, оформить историю коммитов от участников и поставить тег `v1.0` после окончательной проверки.
