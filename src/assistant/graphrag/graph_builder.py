"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)
    edges: dict[tuple[str, str, str], set[str]] = {}
    for result in results:
        for entity in result.entities:
            graph.add_node(entity.node_id)
            data = graph.nodes[entity.node_id]
            data.update(type=entity.type, name=entity.name)
            data.setdefault("sources", set()).add(result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            for entity in (relation.subject, relation.object):
                graph.add_node(entity.node_id)
                data = graph.nodes[entity.node_id]
                data.update(type=entity.type, name=entity.name)
                data.setdefault("sources", set()).add(result.chunk_id)
            edges.setdefault((relation.subject.node_id, relation.type, relation.object.node_id), set()).add(
                result.chunk_id
            )
    for (source, kind, target), sources in edges.items():
        graph.add_edge(source, target, type=kind, weight=len(sources), sources=sorted(sources))
    for node in graph.nodes:
        graph.nodes[node]["sources"] = sorted(graph.nodes[node]["sources"])
    seen: set[frozenset[str]] = set()
    for source, target, data in list(graph.edges(data=True)):
        if data["type"] != "REPLACED_BY":
            continue
        pair = frozenset((source, target))
        if pair in seen or not graph.has_edge(target, source) or graph.edges[target, source]["type"] != "REPLACED_BY":
            continue
        seen.add(pair)
        graph.graph["conflicts"] += 1
        reverse_weight = graph.edges[target, source]["weight"]
        if data["weight"] > reverse_weight:
            graph.remove_edge(target, source)
        elif data["weight"] < reverse_weight:
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.remove_edge(target, source)
    return graph
