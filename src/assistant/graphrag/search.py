"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL = "В базе знаний не найдено."


def _refused() -> Answer:
    return Answer(REFUSAL, refused=True)


def _facts(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> tuple[list[str], list[str]]:
    facts, sources = [], set()
    for source, target in sorted(edges):
        data = graph.edges[source, target]
        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        facts.append(f"{source_name} -{data.get('type', '')}-> {target_name}")
        sources.update(sources_item.rsplit("#", 1)[0] for sources_item in data.get("sources", []))
    return facts, sorted(sources)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    codes = find_codes(question)
    anchors = set()
    for code in codes:
        matches = [node for node, data in graph.nodes(data=True) if data.get("name") == code]
        if not matches:
            return _refused()
        anchors.update(matches)
    for kind, name in find_names(question, kb.vocabulary):
        node = f"{kind}:{name}"
        if node in graph:
            anchors.add(node)
    if not anchors:
        return _refused()
    selected: set[tuple[str, str]] = set()
    if len(anchors) > 1:
        undirected = graph.to_undirected()
        for left, right in combinations(sorted(anchors), 2):
            try:
                paths = nx.all_simple_paths(undirected, left, right, cutoff=3)
                for path in paths:
                    for a, b in zip(path, path[1:], strict=False):
                        selected.add((a, b) if graph.has_edge(a, b) else (b, a))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    else:
        root = next(iter(anchors))
        undirected = graph.to_undirected()
        distances = nx.single_source_shortest_path_length(undirected, root, cutoff=2)
        for node in distances:
            for neighbor in undirected.neighbors(node):
                if neighbor not in distances:
                    continue
                if graph.has_edge(node, neighbor):
                    selected.add((node, neighbor))
                if graph.has_edge(neighbor, node):
                    selected.add((neighbor, node))
    # Include full directed replacement chains reachable from each relevant part.
    parts = {n for edge in selected for n in edge if graph.nodes[n].get("type") == "Part"}
    for part in parts:
        current, seen = part, {part}
        while True:
            next_nodes = [v for _, v, d in graph.out_edges(current, data=True) if d.get("type") == "REPLACED_BY"]
            if not next_nodes:
                break
            nxt = sorted(next_nodes)[0]
            selected.add((current, nxt))
            if nxt in seen:
                break
            seen.add(nxt)
            current = nxt
    fact_list, sources = _facts(graph, selected)
    if not fact_list:
        return _refused()
    facts_text = "\n".join(f"{i}. {fact}" for i, fact in enumerate(fact_list, 1))[:8000]
    answer = llm.chat(
        [
            {
                "role": "user",
                "content": (
                    "Ответь только по фактам ниже. Для цепочки REPLACED_BY назови последнюю деталь; "
                    "не добавляй неподтверждённые сведения.\n"
                    f"Факты:\n{facts_text}\nВопрос: {question}"
                ),
            }
        ]
    ).content.strip()
    return Answer(answer or REFUSAL, sources, fact_list, not bool(answer))


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials, sources = [], set()
    for community in kb.communities:
        if not community.summary:
            continue
        try:
            response = llm.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Ответь только по сводке; если она не помогает, score=0.\n"
                            f"Вопрос: {question}\nСводка: {community.summary}"
                        ),
                    }
                ],
                json_schema=MAP_SCHEMA,
            )
            data = json.loads(response.content)
            answer, score = data["answer"], data["score"]
            if not isinstance(answer, str) or not isinstance(score, int) or not 0 <= score <= 10:
                continue
            if score > 0 and answer.strip():
                partials.append(answer.strip())
                for node in community.nodes:
                    sources.update(s.rsplit("#", 1)[0] for s in kb.graph.nodes[node].get("sources", []))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    if not partials:
        return _refused()
    prompt = "\n".join(
        [
            "Объедини полезные частичные ответы без добавления новых фактов.",
            f"Вопрос: {question}",
            "Частичные ответы:",
            *partials,
        ]
    )
    answer = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    return Answer(answer or REFUSAL, sorted(sources), partials, not bool(answer))
