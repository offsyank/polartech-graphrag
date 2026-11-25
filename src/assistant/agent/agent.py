"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL = "В базе знаний не найдено."
_PART = re.compile(r"\bP-\d{4}[A-Z]?\b", re.IGNORECASE)
_ORDER_ID = re.compile(r"\bSO-\d+\b", re.IGNORECASE)
_PRICE = re.compile(r"(?<!\w)(\d[\d\s\u00a0]*\d|\d)\s*(?:руб(?:лей|ля|ль|\.)?|₽)(?!\w)", re.IGNORECASE)
_QTY = re.compile(r"(?<!\w)(\d+)\s*(?:шт\.?|штук|единиц(?:ы|а)?)(?!\w)", re.IGNORECASE)
_STOCK = re.compile(r"\b(?:остаток|в\s+наличии)\s*[:—-]?\s*(\d+)\b", re.IGNORECASE)

_SYSTEM = (
    "Ты ассистент диспетчера сервисной службы «ПолярТех». Для поиска неисправностей, деталей "
    "и их замен используй search_knowledge, для обзорных вопросов summarize_topic, "
    "для актуальных остатков и цен только check_stock/get_price. Не выдумывай коды, цены и остатки. "
    "Прежде чем вызвать ERP, узнай номер детали из вопроса или графа. "
    "Создание заказа требует явно указанного количества и подтверждения пользователя. "
    "Отличай нулевой остаток от отсутствия данных и недоступности ERP. "
    "При ответе используй только результаты проверенных инструментов."
)


def _key(name: str, args: dict) -> str:
    return json.dumps([name, args], sort_keys=True, ensure_ascii=False, default=str)


def _explicit_quantity(question: str, qty: object) -> bool:
    if not isinstance(qty, int) or isinstance(qty, bool) or qty < 1:
        return False
    patterns = (
        rf"(?<!\w){qty}\s*(?:шт\.?|штук|единиц(?:ы|а)?|детал(?:ь|и|ей)|реле)(?!\w)",
        rf"\b(?:количество|заказать|закажи|заказ|qty)\s*[:—-]?\s*{qty}(?!\d)",
    )
    return any(re.search(pattern, question, re.IGNORECASE) for pattern in patterns)


def _check_arguments(name: str, args: object) -> str | None:
    schema = next((item["function"]["parameters"] for item in TOOL_SCHEMAS if item["function"]["name"] == name), None)
    if schema is None:
        return f"Неизвестный инструмент {name}. Выбери одну из пяти разрешённых функций."
    if not isinstance(args, dict) or "_raw" in args:
        return "Аргументы инструмента не являются корректным JSON-объектом."
    required = set(schema["required"])
    if not required <= args.keys() or set(args) - set(schema["properties"]):
        return "Некорректные аргументы: проверь обязательные поля и их названия."
    for field, value in args.items():
        spec = schema["properties"][field]
        if spec["type"] == "string" and (not isinstance(value, str) or not value.strip()):
            return f"Аргумент {field} должен быть непустой строкой."
        if spec["type"] == "integer" and (type(value) is not int or value < spec.get("minimum", 1)):
            return f"Аргумент {field} должен быть положительным целым числом."
        if "enum" in spec and value not in spec["enum"]:
            return f"Неизвестное значение {field}: {value}."
    return None


def _trusted_values(question: str, steps: list[ToolStep]) -> tuple[set[str], set[int], set[int], set[str]]:
    codes = set(find_codes(question))
    prices = {int(raw.replace(" ", "").replace("\u00a0", "")) for raw in _PRICE.findall(question)}
    quantities = {int(x) for x in _QTY.findall(question)} | {int(x) for x in _STOCK.findall(question)}
    order_ids = set(_ORDER_ID.findall(question.upper()))
    for step in steps:
        if not step.result or "error" in step.result:
            continue
        result = step.result
        if step.tool in {"search_knowledge", "summarize_topic"} and not result.get("refused"):
            raw = " ".join(str(result.get(k, "")) for k in ("answer", "facts"))
            codes.update(find_codes(raw))
        if step.tool in {"get_price", "check_stock", "create_service_order"}:
            codes.update(find_codes(str(result)))
        if step.tool == "get_price" and type(result.get("price_rub")) in (int, float):
            prices.add(int(result["price_rub"]))
        if step.tool == "check_stock" and type(result.get("qty")) is int:
            quantities.add(result["qty"])
        if step.tool == "create_service_order":
            if type(result.get("qty")) is int:
                quantities.add(result["qty"])
            if isinstance(result.get("order_id"), str):
                order_ids.add(result["order_id"].upper())
    return codes, prices, quantities, order_ids


def _unsupported_answer(text: str, question: str, steps: list[ToolStep]) -> str | None:
    codes, prices, quantities, order_ids = _trusted_values(question, steps)
    extra_codes = set(find_codes(text)) - codes
    if extra_codes:
        return f"Неподтверждённые коды: {', '.join(sorted(extra_codes))}."
    unknown_ids = set(_ORDER_ID.findall(text.upper())) - order_ids
    if unknown_ids:
        return "Номер заказа не подтверждён ответом ERP."
    for raw in _PRICE.findall(text):
        if int(raw.replace(" ", "").replace("\u00a0", "")) not in prices:
            return "В ответе указана цена, которой нет в вопросе или результатах ERP."
    for raw in [*_QTY.findall(text), *_STOCK.findall(text)]:
        if int(raw) not in quantities:
            return "В ответе указан неподтверждённый остаток или количество."
    return None


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Обрабатывает вопросы, проверяет источники и запрашивает подтверждение перед заказом."""
    history: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    cache: dict[str, tuple[dict | None, str | None]] = {}
    known_parts = {code for code in find_codes(question) if _PART.fullmatch(code)}
    refusal_seen = False
    tool_steps = 0

    while True:
        response = llm.chat(history, tools=TOOL_SCHEMAS)
        history.append(response.as_message())
        if not response.tool_calls:
            answer = response.content.strip() or REFUSAL
            error = _unsupported_answer(answer, question, steps)
            if error:
                history.append(
                    {"role": "user", "content": f"Исправь итоговый ответ: {error} Не добавляй новых фактов."}
                )
                corrected = llm.chat(history, tools=TOOL_SCHEMAS)
                answer = corrected.content.strip() or REFUSAL
                if corrected.tool_calls or _unsupported_answer(answer, question, steps):
                    return AgentResult(REFUSAL, steps, sorted(sources), refused=True)
            refused = refusal_seen and not any(
                step.tool in {"get_price", "check_stock", "create_service_order"}
                and step.result is not None
                and "error" not in step.result
                for step in steps
            )
            refused = refused or "в базе знаний не найдено" in answer.lower()
            return AgentResult(answer, steps, sorted(sources), refused=refused)

        for call in response.tool_calls:
            if tool_steps >= max(0, max_steps):
                return AgentResult(
                    "Достигнут лимит шагов; работа прервана. Уточните запрос или повторите его.",
                    steps,
                    sorted(sources),
                )
            tool_steps += 1
            name, args = call.name, call.arguments
            step = ToolStep(name, args if isinstance(args, dict) else {})
            error = _check_arguments(name, args)
            if not error and name in {"get_price", "check_stock", "create_service_order"}:
                part_id = args.get("part_id", "")
                if not _PART.fullmatch(part_id) or part_id.upper() not in known_parts:
                    error = (
                        "Номер детали не подтверждён вопросом или результатом поиска. "
                        "Сначала вызови search_knowledge."
                    )
            if not error and name == "create_service_order" and not _explicit_quantity(question, args["qty"]):
                error = "Количество не было явно указано пользователем. Уточни, сколько штук заказать."
            if error:
                step.error = error
                steps.append(step)
                history.append(tool_message(call, {"error": error}))
                continue

            if name == "create_service_order" and not confirm:
                return AgentResult(
                    f"Подготовлен заказ на деталь {args['part_id']} для модели {args['model']}, "
                    f"склад {args['warehouse']}, количество: {args['qty']}. "
                    "Заказ ещё не создан. Подтвердите создание в ERP.",
                    steps,
                    sorted(sources),
                    needs_confirmation=True,
                    pending_action={"tool": name, "args": args},
                )

            key = _key(name, args)
            if key in cache:
                result, error = cache[key]
            else:
                result, error = None, None
                for attempt in range(3):
                    try:
                        result = tools[name](**args)
                        break
                    except ErpUnavailable as exc:
                        if attempt == 2:
                            error = f"Данные ERP получить не удалось: {exc}"
                    except (TypeError, ValueError) as exc:
                        error = f"Ошибка аргументов инструмента: {exc}"
                        break
                if result is not None and "error" in result:
                    error = str(result["error"])
                cache[key] = result, error

            step.result, step.error = result, error
            steps.append(step)
            if name in {"search_knowledge", "summarize_topic"} and result and "error" not in result:
                sources.update(result.get("sources") or [])
                refusal_seen = refusal_seen or bool(result.get("refused"))
                if not result.get("refused"):
                    raw = f"{result.get('answer', '')} {result.get('facts', '')}"
                    known_parts.update(code for code in find_codes(raw) if _PART.fullmatch(code))
            message_data = result if result is not None else {"error": error or "Ошибка инструмента"}
            history.append(tool_message(call, message_data))

        if tool_steps >= max(0, max_steps):
            return AgentResult(
                "Достигнут лимит шагов; работа прервана. Уточните запрос или повторите его.",
                steps,
                sorted(sources),
            )
