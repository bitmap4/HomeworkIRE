# IRE Assignment 1: Indexing and Retrieval System

## Run the Complete Assignment
```bash
docker-compose up app
```

Or run interactively:
```bash
docker-compose run --rm app python main.py
```

## Query Indices

Query Elasticsearch:
```bash
docker-compose run --rm app python query.py es '"machine" AND "learning"'
```

Query custom SelfIndex:
```bash
docker-compose run --rm app python query.py self '("data" OR "science") AND NOT "politics"'
```

## Run Specific Experiments

```bash
docker-compose run --rm app python experiment.py info_comparison
docker-compose run --rm app python experiment.py datastore_comparison
docker-compose run --rm app python experiment.py compression_comparison
docker-compose run --rm app python experiment.py query_proc_comparison
```

## Output

All outputs are mounted as volumes and accessible on your host:

- **plots/** - Frequency distributions, Zipf's law, performance comparisons
- **metrics/** - JSON files with detailed performance metrics
- **indices/** - Persistent storage for custom indices

## Configuration

Edit files in `config/` or override via command line:
```bash
docker-compose run --rm app python main.py data.news.max_docs_per_zip=500
```

---

# Homework for IRE CS4406

Repository of programming and theory homework assignments for IRE course. 
There will also be pointers to other practice work that will not be graded but is made available for those interested in gaining depth and breadth.

## Programming assignments
1. [Indexing and retrieval](https://github.com/CS4406/HomeworkIRE/tree/main/indexing_and_retrieval). This is a programming assignment that will focus on indexing and querying some data.




