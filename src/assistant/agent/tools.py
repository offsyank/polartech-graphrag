"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART_PATTERN = re.compile(r"P-\d{4}[A-Z]?", re.IGNORECASE)
MODEL_PATTERN = re.compile(r"(?:AR|BZ|GS)-\d{3}", re.IGNORECASE)


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_PART = {"type": "string", "description": "Код детали, например P-1042B"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
_QUESTION = {"type": "string", "description": "Вопрос диспетчера"}

TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Найти модель, код ошибки, деталь или замену в техническом графе знаний. Используй до запроса ERP.",
        {"question": _QUESTION},
        ["question"],
    ),
    _schema(
        "summarize_topic",
        "Обобщить сведения по всем документам и сообществам графа знаний.",
        {"question": _QUESTION},
        ["question"],
    ),
    _schema(
        "check_stock",
        "Проверить актуальный остаток детали на конкретном складе ERP.",
        {"part_id": _PART, "warehouse": _WAREHOUSE},
        ["part_id", "warehouse"],
    ),
    _schema(
        "get_price",
        "Получить актуальную цену детали в рублях из ERP.",
        {"part_id": _PART},
        ["part_id"],
    ),
    _schema(
        "create_service_order",
        "Создать заказ в ERP только после явного подтверждения пользователя.",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "description": "Код модели оборудования, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def _part(part_id: object) -> str:
    if not isinstance(part_id, str) or PART_PATTERN.fullmatch(part_id) is None:
        raise ValueError("Некорректный номер детали: нужен код вида P-1042 или P-1042B")
    return part_id.upper()


def _warehouse(warehouse: object) -> str:
    if not isinstance(warehouse, str) or warehouse not in WAREHOUSES:
        raise ValueError("Неизвестный склад; допустимы: " + ", ".join(WAREHOUSES))
    return warehouse


def _quantity(qty: object) -> int:
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        raise ValueError("Количество должно быть положительным целым числом")
    return qty


def _model(model: object) -> str:
    if not isinstance(model, str) or MODEL_PATTERN.fullmatch(model) is None:
        raise ValueError("Некорректный код модели")
    return model.upper()


def _question(question: object) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Вопрос должен быть непустой строкой")
    return question.strip()


def _erp_error(exc: ErpNotFound | ErpRejected) -> dict:
    if isinstance(exc, ErpNotFound):
        return {"error": f"Данных в ERP нет: {exc}"}
    return {"error": f"ERP отклонила операцию: {exc}"}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Возвращает пять проверяющих аргументы инструментов для работы с графом и ERP."""

    def search_knowledge(question: str) -> dict:
        try:
            result = local_search(_question(question), kb, llm)
        except ValueError as exc:
            return {"error": str(exc)}
        return {"answer": result.text, "facts": result.facts, "sources": result.sources, "refused": result.refused}

    def summarize_topic(question: str) -> dict:
        try:
            result = global_search(_question(question), kb, llm)
        except ValueError as exc:
            return {"error": str(exc)}
        return {"answer": result.text, "sources": result.sources, "refused": result.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        try:
            validated_part = _part(part_id)
            validated_warehouse = _warehouse(warehouse)
            return erp.get_stock(validated_part, validated_warehouse)
        except ErpNotFound as exc:
            return _erp_error(exc)
        except ErpRejected as exc:
            return _erp_error(exc)
        except ValueError as exc:
            return {"error": str(exc)}

    def get_price(part_id: str) -> dict:
        try:
            validated_part = _part(part_id)
            return erp.get_price(validated_part)
        except ErpNotFound as exc:
            return _erp_error(exc)
        except ErpRejected as exc:
            return _erp_error(exc)
        except ValueError as exc:
            return {"error": str(exc)}

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        try:
            validated_part = _part(part_id)
            validated_warehouse = _warehouse(warehouse)
            validated_quantity = _quantity(qty)
            validated_model = _model(model)
            return erp.create_order(validated_part, validated_warehouse, validated_quantity, validated_model)
        except ErpNotFound as exc:
            return _erp_error(exc)
        except ErpRejected as exc:
            return _erp_error(exc)
        except ValueError as exc:
            return {"error": str(exc)}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
