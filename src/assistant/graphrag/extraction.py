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
    codes = set(find_codes(chunk.text))
    names = {kind: set(values) for kind, values in vocab.items()}
    prompt = (
        "Извлеки только прямо подтверждённые текстом сущности и связи по указанной схеме. "
        "Направление важно: Model -> Part — HAS_PART; Part -> Component — IS_A; "
        "ErrorCode -> Component — INDICATES; Part -> Part — REPLACED_BY. "
        "Пример: для «P-1 заменена на P-2» верни subject=P-1, type=REPLACED_BY, object=P-2.\n\n"
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
