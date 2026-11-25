"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    vocab = vocabulary()
    code_list = find_codes(chunk.text)
    codes = set(code_list)
    names = {kind: set(values) for kind, values in vocab.items()}
    part_example = (
        ""
        if "## Коды ошибок" in chunk.text and "## Состав узлов" not in chunk.text
        else (
            "В HAS_PART поле object ВСЕГДА содержит номер P-…, НЕ название узла. "
            "Например, для «компрессор — деталь P-1000» в модели AR-100 нужны две разные связи: "
            "AR-100 HAS_PART P-1000; P-1000 IS_A компрессор. "
        )
    )
    prompt = (
        "Извлеки подтверждённые текстом сущности и связи. Поле subject — начало связи, object — конец. "
        "Имя должно быть кодом без слов перед ним или названием из словаря. Part — только номер детали P-…, "
        "Component — название узла, даже если в тексте его называют деталью. Типы и направление обязательны:\n"
        "BELONGS_TO: subject Model, object ProductLine — модель входит в линейку.\n"
        "HAS_PART: subject Model, object Part — модель содержит указанную деталь.\n"
        "FAILED: subject Model, object Component — у модели отказал узел по отчёту о выезде.\n"
        "IS_A: subject Part, object Component — номер детали соответствует названию узла.\n"
        "REPLACED_BY: subject Part (старая, снятая с производства), object Part (новая замена).\n"
        "SUPPLIED_BY: subject Part, object Supplier — поставщик указан для детали.\n"
        "INDICATES: subject ErrorCode, object Component — код ошибки указывает на отказ узла.\n"
        "FIXED_BY: subject ErrorCode, object Procedure — регламент применяется к ошибке.\n"
        "В таблице состава узлов извлекай HAS_PART, IS_A и SUPPLIED_BY для каждой строки. "
        f"{part_example}"
        "Отчёт о выезде подтверждает FAILED; наличие узла не означает отказ.\n"
        "Пример формата для «AR-100 содержит деталь P-1000»: "
        '{"entities":[{"type":"Model","name":"AR-100"},{"type":"Part","name":"P-1000"}],'
        '"relations":[{"subject_type":"Model","subject":"AR-100","type":"HAS_PART",'
        '"object_type":"Part","object":"P-1000"}]}. '
        'Для «P-1000 снята с производства, вместо неё ставят P-1000B» начало связи — P-1000, конец — P-1000B.\n'
        f"Коды из текста: {', '.join(code_list)}\n"
        f"Словарь: {json.dumps(vocab, ensure_ascii=False)}\n\n"
        f"Текст чанка:\n{chunk.text}"
    )
    empty = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    for attempt in range(2):
        try:
            response = llm.chat(
                [
                    {
                        "role": "user",
                        "content": prompt if attempt == 0 else prompt + "\nВерни корректный JSON строго по схеме.",
                    }
                ],
                json_schema=EXTRACTION_SCHEMA,
                max_tokens=2600,
            )
            data = json.loads(response.content)
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("entities"), list)
                or not isinstance(data.get("relations"), list)
            ):
                raise ValueError("invalid extraction shape")
            entities: dict[str, Entity] = {}
            dropped = 0
            for raw in data["entities"]:
                if (
                    not isinstance(raw, dict)
                    or not isinstance(raw.get("type"), str)
                    or not isinstance(raw.get("name"), str)
                ):
                    raise ValueError("invalid entity")
                kind, value = raw["type"], raw["name"]
                if kind in {"Model", "Part", "ErrorCode", "Procedure"}:
                    code = normalize_code(value)
                    canonical = code if code and code_type(code) == kind and code in codes else None
                else:
                    canonical = canonical_name(kind, value, vocab)
                    if (
                        canonical is None
                        or canonical not in names.get(kind, set())
                        or canonical.lower() not in chunk.text.lower()
                    ):
                        canonical = None
                if canonical is None:
                    dropped += 1
                else:
                    entity = Entity(kind, canonical)
                    entities[entity.node_id] = entity
            relations = []
            for raw in data["relations"]:
                if not isinstance(raw, dict):
                    raise ValueError("invalid relation")

                def endpoint(kind: str, value: str) -> Entity | None:
                    if kind in {"Model", "Part", "ErrorCode", "Procedure"}:
                        code = normalize_code(value)
                        canonical = code if code and code_type(code) == kind and code in codes else None
                    else:
                        canonical = canonical_name(kind, value, vocab)
                        if (
                            canonical is None
                            or canonical not in names.get(kind, set())
                            or canonical.lower() not in chunk.text.lower()
                        ):
                            canonical = None
                    return Entity(kind, canonical) if canonical else None

                subject = endpoint(raw["subject_type"], raw["subject"])
                obj = endpoint(raw["object_type"], raw["object"])
                if subject and obj:
                    entities[subject.node_id] = subject
                    entities[obj.node_id] = obj
                    relations.append(Relation(subject, raw["type"], obj))
                else:
                    dropped += 1
            return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), relations, dropped)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            if attempt == 1:
                return empty
    return empty
