# Wikipedia PageRank & Personalized PageRank (Nantes) — PySpark DataFrame

> **Cours :** Large Scale Data Management (2026–2027) — Pascal Molli  
> **<Travail en solo>:**
> - Méwen GRAIS*


---

## 1. Objectif & contraintes techniques

Ce projet implémente, uniquement avec l'API PySpark DataFrame (aucun appel `.rdd`, aucune bibliothèque GraphFrames), un moteur unique et paramétrable calculant sur le graphe des liens de Wikipedia anglophone (*WikiLinkGraphs*, 1er mars 2018) :

1. **PageRank classique (PR)** : $v(p) = 1/N$ pour toute page $p \in V$ ($N = |\text{sources} \cup \text{cibles}|$).
2. **PageRank personnalisé (PPR) depuis Nantes (`page_id = 67453`)** : $v(67453) = 1$, $v(p) = 0$ ailleurs.

À chaque itération ($d = 0{,}85$, initialisation $r^{(0)} = v$, exactement 20 itérations) :

$$r^{(t+1)}(p) = \bigl(1 - d + d\,D^{(t)}\bigr)\,v(p) + d \sum_{u \to p} \frac{r^{(t)}(u)}{\deg(u)}$$

où $D^{(t)} = \sum_{u : \deg(u) = 0} r^{(t)}(u)$ est la masse des *dangling nodes* redistribuée selon $v$.  
L'invariant $\sum_{p \in V} r^{(t)}(p) = 1 \pm 10^{-6}$ est vérifié automatiquement à chaque itération.

---

## 2. Architecture du projet

```text
.
├── README.md                        # Rapport final (résultats, temps, plan explain, analyse)
├── pyproject.toml                   # Dépendances (pyspark 3.5, networkx, scipy, pandas, pytest)
├── src/
│   ├── config.py                    # Constantes du sujet et configuration Spark (AQE=false, K=96)
│   ├── data_prep.py                 # Stratégie anti-.csv.gz (Parquet partitionné) & sample induit 10 %
│   ├── pagerank.py                  # Moteur unifié PR/PPR co-partitionné + localCheckpoint + garde-fou
│   ├── analysis.py                  # Top 20 PR, Top 20 PPR, diagnostic multi-seuils, Top 20 PPR/PR
│   └── main.py                      # Orchestrateur CLI (`prepare`, `sample`, `run`) & chronométrage
├── tests/
│   └── test_pagerank_networkx.py    # Validation unitaire PR & PPR contre networkx.pagerank
├── scripts/
│   ├── download_data.sh             # Téléchargement streaming Zenodo -> GCS (2018 & bonus 2008)
│   ├── create_cluster.sh            # Création cluster Dataproc 2, 4 ou 6 workers n2-standard-4 (<= 32 vCPU)
│   └── submit_job.sh                # Soumission spark-submit avec src.zip et tuning mémoire/AQE
└── results/
    ├── explain_iteration.txt        # Preuve du plan d'exécution avec 1 seul Exchange actif
    ├── benchmarks.csv               # Temps de préparation vs temps par itération (2, 4, 6 nœuds)
    ├── top20_pr.csv                 # Top 20 PageRank classique (2018)
    ├── top20_ppr_nantes.csv         # Top 20 PPR depuis Nantes (2018)
    ├── ppr_threshold_diagnostics.csv# Sensibilité multi-seuils pour justifier le seuil PPR/PR
    └── top20_ratio_ppr_pr.csv       # Top 20 du ratio PPR(p) / PR(p) filtré
```

### Validation locale (avant passage à l'échelle)
```bash
pytest tests/test_pagerank_networkx.py
```
- **Résultat** : `3 passed` (écart `< 1e-10` à 20 itérations et `< 1e-6` à convergence face à `networkx.pagerank` en PR et PPR avec *dangling nodes*, `sum(r) == 1.0` vérifié à chaque itération, et `exchange_count == 1`).

---

## 3. Top 20 du PageRank classique (2018)

> **Question :** Quelle est l'entité au centre de Wikipedia en 2018 ?

*(À remplir depuis `results/top20_pr.csv` après exécution Dataproc sur 100 % des données 2018.)*

| Rang | `page_id` | Titre Wikipedia | `PR(p)` |
| ---: | ---: | :--- | ---: |
| 1 | | | |
| 2 | | | |
| ... | | | |
| 20 | | | |

**Analyse :** *(Commenter l'entité centrale de Wikipedia en 2018 — pays, grandes zones géographiques, entités transverses — et expliquer pourquoi elles concentrent le PageRank global).*

---

## 4. Top 20 du PPR depuis Nantes (`67453`) & Top 20 du ratio `PPR(p) / PR(p)`

### 4.1. Top 20 du PPR brut depuis Nantes (`results/top20_ppr_nantes.csv`)

| Rang | `page_id` | Titre Wikipedia | `PPR(p)` |
| ---: | ---: | :--- | ---: |
| 1 | 67453 | Nantes | |
| 2 | | | |
| ... | | | |
| 20 | | | |

### 4.2. Choix et justification du seuil minimal de PPR (`results/ppr_threshold_diagnostics.csv`)

Sans seuil minimal, les pages périphériques ayant un `PR(p)` très proche du plancher $(1 - d)/N$ mais un lien indirect depuis Nantes produisent un rapport `PPR(p) / PR(p)` artificiellement gonflé par un dénominateur quasi nul.

*(À remplir depuis `results/ppr_threshold_diagnostics.csv`.)*

| Seuil candidat | Valeur | Pages retenues | % du graphe | Masse PPR couverte | Ratio max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `1/N (uniforme)` | | | | | |
| `5/N` | | | | | |
| `10/N` | | | | | |
| `1e-5` | | | | | |
| `1e-4` | | | | | |

**Seuil retenu et justification :** *(Justifier le seuil choisi — par défaut `10/N` ou `1e-5` — à partir du compromis entre élimination du bruit de queue et conservation de la majorité de la masse PPR).*

### 4.3. Top 20 du ratio `PPR(p) / PR(p)` filtré (`results/top20_ratio_ppr_pr.csv`)

| Rang | `page_id` | Titre Wikipedia | `PPR(p)` | `PR(p)` | `PPR / PR` |
| ---: | ---: | :--- | ---: | ---: | ---: |
| 1 | | | | | |
| 2 | | | | | |
| ... | | | | | |
| 20 | | | | | |

### 4.4. Comparaison PR vs PPR brut vs `PPR / PR`
*(Commenter les différences : le PPR brut reste dominé par les hubs mondiaux/nationaux comme France, Europe, Paris ; le ratio `PPR / PR` neutralise la centralité globale et fait émerger les entités spécifiquement nantaises — Loire-Atlantique, Pays de la Loire, FC Nantes, Université de Nantes, Jules Verne, etc.).*

---

## 5. Benchmarks : 2, 4 et 6 nœuds (Google Cloud Dataproc)

### Protocole & matériel (strictement identique par nœud, ≤ 32 vCPU)
- **Plateforme** : Google Cloud Dataproc `2.2-debian12` (Apache Spark 3.5.x, Python 3.11) + Google Cloud Storage (`gs://`).
- **Matériel par nœud** : `n2-standard-4` (4 vCPU, 16 Go RAM, 50 Go `pd-balanced`).
- **Topologies testées** :
  - **2 workers** (+ 1 master) = 12 vCPU ≤ 32 vCPU
  - **4 workers** (+ 1 master) = 20 vCPU ≤ 32 vCPU
  - **6 workers** (+ 1 master) = 28 vCPU ≤ 32 vCPU
- **Configuration Spark fixe** : `spark.sql.shuffle.partitions = 96` (multiple commun de 8, 16 et 24 cœurs), `spark.sql.adaptive.enabled = false`, `spark.sql.autoBroadcastJoinThreshold = -1`, `spark.executor.cores = 4`, `spark.executor.memory = 10g`, `spark.executor.memoryOverhead = 2g`.

### Phase 0 — Stratégie anti-`.csv.gz` (exécutée 1 seule fois)
Un fichier `.csv.gz` est non découpable (*non-splittable*) et lu sur 1 seul cœur. Nous le lisons **une seule fois** avec schéma explicite (`WIKILINK_RAW_SCHEMA`, sans `inferSchema=True`), nettoyons les doublons/auto-boucles, et écrivons `edges.parquet` + `titles.parquet` en Parquet Snappy partitionné en 96 fichiers.
- **Temps de conversion `.csv.gz` → Parquet (2018)** : *(à remplir)* s.

### Tableau des temps d'exécution (`results/benchmarks.csv`, 100 % dataset 2018)

| Workers (`n2-standard-4`) | vCPU workers | Préparation graphe (Parquet + co-partitionnement `K=96`) | PR total (20 iter.) | PR moy. / iter. | PPR total (20 iter.) | PPR moy. / iter. | Speedup (vs 2w) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **2** | 8 | s | s | s | s | s | 1.00× |
| **4** | 16 | s | s | s | s | s | × |
| **6** | 24 | s | s | s | s | s | × |

**Analyse de la scalabilité :** *(Commenter le passage de 2 à 4 puis 6 workers : gain quasi-linéaire sur les itérations grâce à l'absence de shuffle sur les jointures, coût résiduel du shuffle unique du `groupBy`, et comparaison PR vs PPR).*

---

## 6. Preuve du plan d'exécution (`.explain("formatted")`) & coupure de lignée

Le plan complet d'une itération est disponible dans [`results/explain_iteration.txt`](results/explain_iteration.txt).

### Structure physique active de l'itération 2
```text
== Physical Plan ==
* Project                                          <-- r(p) = (1 - d + d*D)*v(p) + d*coalesce(incoming, 0)
+- * SortMergeJoin LeftOuter                       <-- [0 Exchange] nodes ⋈_left contributions (co-partitionnés sur K)
   :- InMemoryTableScan                            <-- `base_nodes` en cache (HashPartitioning(page_id, K))
   +- * Sort                                       <-- Tri local par partition uniquement
      +- * HashAggregate                           <-- sum(contrib) final par destination
         +- Exchange hashpartitioning(page_id, K)  <-- *** L'UNIQUE EXCHANGE ACTIF DE L'ITÉRATION ***
            +- * HashAggregate                     <-- partial_sum(contrib) local (combiner avant shuffle)
               +- * Project                        <-- contrib = rank / out_degree
                  +- * SortMergeJoin Inner         <-- [0 Exchange, 0 Sort] links ⋈ ranks (co-partitionnés sur K)
                     :- InMemoryTableScan          <-- `links` en cache (HashPartitioning(page_id, K))
                     +- * Project
                        +- * Scan ExistingRDD      <-- `ranks` issu de localCheckpoint(eager=True)
```

### Pourquoi il n'y a qu'un seul `Exchange` par itération
1. **Co-partitionnement statique (`K = 96`)** : avant la boucle, `links` et `base_nodes` sont hash-partitionnés sur `page_id` en exactement `K` partitions (`repartition(K, "page_id").persist(MEMORY_AND_DISK)`).
2. **Coupure de lignée par `localCheckpoint(eager=True)`** : appelé à la fin de chaque itération, `localCheckpoint(eager=True)` tronque le DAG Catalyst (remplacé par un unique `Scan ExistingRDD`) tout en **préservant** dans `LogicalRDD` les métadonnées `hashpartitioning(page_id, K)` et le tri `[page_id ASC NULLS FIRST]`. La jointure `links ⋈ ranks` s'exécute donc avec **0 `Exchange` et 0 `Sort`**.
3. **Neutralisation de l'AQE (`spark.sql.adaptive.enabled = false`)** : avec l'AQE activé, Spark peut fusionner dynamiquement les partitions (`CoalesceShufflePartitions`) après le `groupBy`, produisant `K' != K` partitions et cassant l'alignement avec `links` et `base_nodes`. Désactiver l'AQE + fixer `spark.sql.autoBroadcastJoinThreshold = -1` garantit un plan déterministe à **1 seul `Exchange`** (vérifié automatiquement par le garde-fou `count_exchanges_in_plan`).

*(Note : les blocs `InMemoryRelation` imbriqués sous `InMemoryTableScan` affichent le plan historique qui a construit le cache avant la boucle ; ils ne sont pas ré-exécutés durant les itérations).*

---

## 7. Bonus — Évolution dans le temps : 2008 vs 2018

*(Relancer le même pipeline avec `--raw-input gs://BUCKET/raw/enwiki.wikilink_graph.2008-03-01.csv.gz` et `--dataset-label full_2008`.)*

| Critère | Instantané 2008 | Instantané 2018 | Évolution observée |
| :--- | :--- | :--- | :--- |
| Nombre de pages ($N$) | | | |
| Nombre de liens ($M$) | | | |
| Top 5 PR (centre de Wikipedia) | | | |
| Top 5 ratio `PPR / PR` (Nantes) | | | |

**Commentaire :** *(Analyser si le centre de Wikipedia et le voisinage spécifique de Nantes ont évolué en 10 ans).*

---

## 8. Références

- S. Brin, L. Page. *The Anatomy of a Large-Scale Hypertextual Web Search Engine*. Computer Networks and ISDN Systems 30(1-7), 1998.
- L. Page, S. Brin, R. Motwani, T. Winograd. *The PageRank Citation Ranking: Bringing Order to the Web*. Stanford InfoLab, 1999.
- T. H. Haveliwala. *Topic-Sensitive PageRank*. WWW 2002.
- D. F. Gleich. *PageRank Beyond the Web*. SIAM Review 57(3), 2015.
- M. Zaharia et al. *Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing*. NSDI 2012.
- M. Armbrust et al. *Spark SQL: Relational Data Processing in Spark*. SIGMOD 2015.
- C. Consonni, D. Laniado, A. Montresor. *WikiLinkGraphs: A Complete, Longitudinal and Multi-Language Dataset of the Wikipedia Link Networks*. ICWSM 2019 (`doi:10.5281/zenodo.2539424`).
