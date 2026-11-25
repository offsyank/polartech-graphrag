"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx
from networkx.algorithms.community import louvain_communities

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if not graph:
        return []
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    groups = louvain_communities(undirected, weight="weight", seed=7)
    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: nodes[0])
    return [Community(i, nodes) for i, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    nodes = set(community.nodes)
    details = []
    counts: dict[str, int] = {}
    for source, target, data in graph.edges(data=True):
        if source not in nodes or target not in nodes:
            continue
        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        weight = data.get("weight", 1)
        details.append(f"{source_name} -{data.get('type', '')}-> {target_name} (вес {weight})")
        if data.get("type") == "FAILED":
            counts[target_name] = counts.get(target_name, 0) + weight
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    prompt = (
        "Составь краткую сводку сообщества на русском, не более 150 слов. Опиши связи и наиболее "
        "часто отказывающий узел, если он есть. Не придумывай числа; используй переданную статистику.\n"
        f"Сущности: {', '.join(graph.nodes[n].get('name', n) for n in community.nodes)}\n"
        f"Связи: {'; '.join(details)}\nЧастота отказов по весу FAILED: {ranked}"
    )
    summary = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    words = summary.split()
    return " ".join(words[:150])
