"""Tests unitaires validant le moteur PySpark DataFrame contre NetworkX.

On vérifie sur un mini-graphe dirigé contenant :
- des cycles,
- des pages sans lien sortant (dangling nodes),
- une page sans lien entrant (source pure),
que :
1. Le PageRank classique (PR) correspond exactement à `networkx.pagerank` (tolérance 1e-6).
2. Le PageRank personnalisé (PPR) correspond exactement à `networkx.pagerank(..., personalization=...)`.
3. La somme des rangs reste égale à 1.0 à chaque itération.
4. Le plan d'exécution `.explain('formatted')` ne contient qu'un seul `Exchange` par itération.
5. Une page source PPR absente du graphe lève bien une `ValueError`.
"""

from __future__ import annotations

import networkx as nx
import pytest
from pyspark.sql import SparkSession

from src.config import (
    COL_DST,
    COL_PAGE_ID,
    COL_RANK,
    COL_SRC,
    DAMPING_FACTOR,
    LOCAL_TEST_NUM_PARTITIONS,
    RANK_SUM_TOLERANCE,
    SparkConfig,
    build_spark_session,
)
from src.pagerank import prepare_graph_tables, run_pagerank


# Mini-graphe dirigé de test (6 sommets : 1..6)
# - Sommet 4 : dangling node (aucun lien sortant)
# - Sommet 6 : dangling node (aucun lien sortant)
# - Sommet 5 : aucun lien entrant (doit rester présent grâce au left_outer join)
# - Cycle 1 <-> 2 -> 3 -> 1 + liens vers les dangling nodes 4 et 6
TEST_EDGES: list[tuple[int, int]] = [
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 3),
    (3, 1),
    (3, 4),
    (3, 6),
    (5, 2),
    (5, 6),
]

TEST_NODES = sorted({u for u, _ in TEST_EDGES} | {v for _, v in TEST_EDGES})


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Session Spark locale configurée exactement comme en production (AQE=false, broadcast=-1)."""
    session = build_spark_session(
        SparkConfig(
            app_name="Test-PageRank-NetworkX",
            num_partitions=LOCAL_TEST_NUM_PARTITIONS,
            master="local[2]",
            disable_aqe=True,
            disable_auto_broadcast=True,
        )
    )
    yield session
    session.stop()


def _make_edges_df(spark: SparkSession, edges: list[tuple[int, int]]):
    """Construit un DataFrame `(src, dst)` nativement côté JVM (sans worker Python)."""
    values_sql = ", ".join(f"({int(u)}, {int(v)})" for u, v in edges)
    return spark.sql(
        f"SELECT CAST(col1 AS INT) AS {COL_SRC}, CAST(col2 AS INT) AS {COL_DST} "
        f"FROM VALUES {values_sql}"
    )


def _nx_power_iteration(
    graph: nx.DiGraph,
    alpha: float,
    num_iterations: int,
    personalization: dict[int, float] | None = None,
) -> dict[int, float]:
    """Référence exacte de power iteration à nombre fixe d'itérations (pour valider 20 itérations exactes)."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if personalization is None:
        v = {u: 1.0 / n for u in nodes}
    else:
        v = {u: float(personalization.get(u, 0.0)) for u in nodes}

    r = dict(v)
    dangling_nodes = [u for u in nodes if graph.out_degree(u) == 0]

    for _ in range(num_iterations):
        dangling_sum = sum(r[u] for u in dangling_nodes)
        new_r = {u: (1.0 - alpha + alpha * dangling_sum) * v[u] for u in nodes}
        for u in nodes:
            deg = graph.out_degree(u)
            if deg > 0:
                share = alpha * r[u] / deg
                for nbr in graph.successors(u):
                    new_r[nbr] += share
        r = new_r
    return r


def test_classic_pagerank_matches_networkx(spark: SparkSession) -> None:
    """Valide PR classique (20 itérations + convergence contre networkx.pagerank) et 1 seul Exchange."""
    edges_df = _make_edges_df(spark, TEST_EDGES)
    links_df, base_nodes_df, num_nodes, num_edges, num_dangling = prepare_graph_tables(
        edges_df, num_partitions=LOCAL_TEST_NUM_PARTITIONS
    )

    assert num_nodes == 6
    assert num_edges == len(TEST_EDGES)
    assert num_dangling == 2  # Sommets 4 et 6

    # 1. Validation exacte à 20 itérations (critère du sujet)
    res_20 = run_pagerank(
        links_df=links_df,
        base_nodes_df=base_nodes_df,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_dangling=num_dangling,
        source_page_id=None,
        damping_factor=DAMPING_FACTOR,
        num_iterations=20,
        enforce_single_exchange=True,
    )

    assert res_20.mode == "pr"
    assert res_20.exchange_count == 1
    assert len(res_20.rank_sums) == 20
    assert all(abs(s - 1.0) <= RANK_SUM_TOLERANCE for s in res_20.rank_sums)

    g = nx.DiGraph()
    g.add_edges_from(TEST_EDGES)

    spark_pr_20 = {
        int(row[COL_PAGE_ID]): float(row[COL_RANK]) for row in res_20.ranks.collect()
    }
    ref_pr_20 = _nx_power_iteration(g, alpha=DAMPING_FACTOR, num_iterations=20)

    assert set(spark_pr_20.keys()) == set(TEST_NODES)
    for node in TEST_NODES:
        assert spark_pr_20[node] == pytest.approx(ref_pr_20[node], abs=1e-10)

    # 2. Validation à convergence directe contre `networkx.pagerank` (60 itérations)
    res_conv = run_pagerank(
        links_df=links_df,
        base_nodes_df=base_nodes_df,
        num_nodes=num_nodes,
        source_page_id=None,
        damping_factor=DAMPING_FACTOR,
        num_iterations=60,
        enforce_single_exchange=True,
    )
    spark_pr_conv = {
        int(row[COL_PAGE_ID]): float(row[COL_RANK]) for row in res_conv.ranks.collect()
    }
    nx_pr = nx.pagerank(g, alpha=DAMPING_FACTOR, tol=1e-12, max_iter=500)

    for node in TEST_NODES:
        assert spark_pr_conv[node] == pytest.approx(nx_pr[node], abs=1e-6)

    links_df.unpersist()
    base_nodes_df.unpersist()


def test_personalized_pagerank_matches_networkx(spark: SparkSession) -> None:
    """Valide PPR depuis un sommet cible (incluant redistribution des dangling nodes vers la source)."""
    edges_df = _make_edges_df(spark, TEST_EDGES)
    links_df, base_nodes_df, num_nodes, num_edges, num_dangling = prepare_graph_tables(
        edges_df, num_partitions=LOCAL_TEST_NUM_PARTITIONS
    )

    source_node = 1
    personalization = {u: (1.0 if u == source_node else 0.0) for u in TEST_NODES}

    g = nx.DiGraph()
    g.add_edges_from(TEST_EDGES)

    # 1. Exactitude à 20 itérations
    res_20 = run_pagerank(
        links_df=links_df,
        base_nodes_df=base_nodes_df,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_dangling=num_dangling,
        source_page_id=source_node,
        damping_factor=DAMPING_FACTOR,
        num_iterations=20,
        enforce_single_exchange=True,
    )

    assert res_20.mode == "ppr"
    assert res_20.exchange_count == 1
    assert all(abs(s - 1.0) <= RANK_SUM_TOLERANCE for s in res_20.rank_sums)

    spark_ppr_20 = {
        int(row[COL_PAGE_ID]): float(row[COL_RANK]) for row in res_20.ranks.collect()
    }
    ref_ppr_20 = _nx_power_iteration(
        g, alpha=DAMPING_FACTOR, num_iterations=20, personalization=personalization
    )

    for node in TEST_NODES:
        assert spark_ppr_20[node] == pytest.approx(ref_ppr_20[node], abs=1e-10)
    # Le sommet 5 n'a aucun lien entrant et v(5) = 0 -> son PPR doit être exactement 0.0
    assert spark_ppr_20[5] == pytest.approx(0.0, abs=1e-12)

    # 2. Validation à convergence contre `networkx.pagerank(..., personalization=..., dangling=...)`
    res_conv = run_pagerank(
        links_df=links_df,
        base_nodes_df=base_nodes_df,
        num_nodes=num_nodes,
        source_page_id=source_node,
        damping_factor=DAMPING_FACTOR,
        num_iterations=60,
        enforce_single_exchange=True,
    )
    spark_ppr_conv = {
        int(row[COL_PAGE_ID]): float(row[COL_RANK]) for row in res_conv.ranks.collect()
    }
    # Dans NetworkX, `dangling` par défaut vaut `personalization` (redistribution selon v)
    nx_ppr = nx.pagerank(
        g,
        alpha=DAMPING_FACTOR,
        personalization=personalization,
        dangling=personalization,
        tol=1e-12,
        max_iter=500,
    )

    for node in TEST_NODES:
        assert spark_ppr_conv[node] == pytest.approx(nx_ppr[node], abs=1e-6)

    links_df.unpersist()
    base_nodes_df.unpersist()


def test_ppr_missing_source_raises_value_error(spark: SparkSession) -> None:
    """Vérifie qu'un identifiant source absent du graphe est rejeté immédiatement."""
    edges_df = _make_edges_df(spark, TEST_EDGES)
    links_df, base_nodes_df, num_nodes, _, _ = prepare_graph_tables(
        edges_df, num_partitions=LOCAL_TEST_NUM_PARTITIONS
    )

    with pytest.raises(ValueError, match="absente du graphe"):
        run_pagerank(
            links_df=links_df,
            base_nodes_df=base_nodes_df,
            num_nodes=num_nodes,
            source_page_id=999999,
            num_iterations=2,
        )

    links_df.unpersist()
    base_nodes_df.unpersist()
