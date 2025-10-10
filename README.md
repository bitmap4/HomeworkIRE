# IRE Assignment 1: Indexing and Retrieval System

All project files are located in the `indexing_and_retrieval/` directory.

## Run the Complete Assignment

First download the [news dataset](https://github.com/Webhose/free-news-datasets) and the [wiki dataset](https://huggingface.co/datasets/wikimedia/wikipedia) and put them in the `indexing_and_retrieval/data/` directory in this format:
```
indexing_and_retrieval/data/
├── free-news-datasets/
└── 20231101.en/
```
For the configuration I have used, the wiki data is not needed. This can be modified in `indexing_and_retrieval/config/config.yaml`:
```yaml
paths:
  news_dir: indexing_and_retrieval/data/free-news-datasets/News_Datasets
  wiki_file: indexing_and_retrieval/data/20231101.en/train-00006-of-00041.parquet
```

Then run:
```bash
cd indexing_and_retrieval
docker-compose up app
```

Or run interactively:
```bash
cd indexing_and_retrieval
docker-compose run --rm app python scripts/main.py
```

## Query Indices

Query Elasticsearch:
```bash
cd indexing_and_retrieval
docker-compose run --rm app python scripts/query.py es '"machine" AND "learning"'
```

Query custom SelfIndex:
```bash
cd indexing_and_retrieval
docker-compose run --rm app python scripts/query.py self '("data" OR "science") AND NOT "politics"'
```

## Run Specific Experiments

```bash
cd indexing_and_retrieval
docker-compose run --rm app python scripts/experiment.py info_comparison
docker-compose run --rm app python scripts/experiment.py datastore_comparison
docker-compose run --rm app python scripts/experiment.py compression_comparison
docker-compose run --rm app python scripts/experiment.py query_proc_comparison
```

## Output

All outputs are available in the `indexing_and_retrieval/outputs/` directory:

- **plots/** - Frequency distributions, Zipf's law, performance comparisons
- **metrics/** - JSON files with detailed performance metrics
- **indices/** - Persistent storage for custom indices

## Configuration

Edit files in `indexing_and_retrieval/config/` or override via command line:
```bash
cd indexing_and_retrieval
docker-compose run --rm app python scripts/main.py data.news.max_docs_per_zip=500
```

---

# Homework for IRE CS4406

Repository of programming and theory homework assignments for IRE course. 
There will also be pointers to other practice work that will not be graded but is made available for those interested in gaining depth and breadth.

## Programming assignments
1. [Indexing and retrieval](https://github.com/CS4406/HomeworkIRE/tree/main/indexing_and_retrieval). This is a programming assignment that will focus on indexing and querying some data.




