# Bluesky Data Analyzer

A comprehensive pipeline for collecting, processing, and analyzing Bluesky social media data. This project enables you to download posts based on keywords, filter and clean the data, perform semantic analysis, clustering, and generate visualizations.

## Overview

The project consists of several Python scripts that work together to:

1. **Download** Bluesky posts matching specified keywords and date ranges
2. **Filter** posts by engagement metrics and content quality
3. **Enrich** reply data with parent/root post context
4. **Extract** semantic content for analysis
5. **Embed** content using OpenAI's text-embedding-3-large model
6. **Analyze** posting patterns and engagement metrics
7. **Cluster** posts based on semantic similarity
8. **Visualize** results through interactive dashboards

## Requirements

- Python 3.8+
- OpenAI API key (for embeddings)
- Required packages:
  - pandas
  - numpy
  - plotly
  - umap-learn
  - hdbscan
  - scikit-learn
  - openai
  - python-dotenv

## Setup

1. **Install dependencies** (recommended in a virtual environment):
   ```bash
   pip install pandas numpy plotly umap-learn hdbscan scikit-learn openai python-dotenv
   ```

2. **Create a `.env` file** with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Configure your analysis** by editing `config.json`:
   - Set your desired date range
   - Specify keywords to search for
   - Adjust filtering parameters (minimum likes, etc.)

## Configuration File

The `config.json` file controls the data collection parameters:

```json
{
  "run_name": "august",
  "date_range": {
    "start_date": "2025-08-01",
    "end_date": "2025-08-31"
  },
  "keywords": [
    "ai", "chatgpt", "llm", "openai", "claude", ...
  ],
  "filters": {
    "min_likes": 10
  }
}
```

- **run_name**: Identifier for this analysis session (used in output filenames)
- **date_range**: Start and end dates for data collection
- **keywords**: List of terms to search for in posts
- **filters.min_likes**: Minimum number of likes required for a post to be included

## Usage Instructions

### 1. Main Pipeline (Complete Analysis)

Run the full pipeline from data collection to processing:

```bash
python main_pipeline.py
```

This executes all pipeline steps sequentially:
- Downloads raw data from Bluesky API
- Filters posts with minimum likes
- Filters out spam and adult content
- Enriches replies with parent/root context
- Extracts semantic content for embeddings

### 2. Content Embedding

Generate semantic embeddings for the processed data:

```bash
python embed_content.py <session_name>
```

**Example:**
```bash
python embed_content.py august
```

**Options:**
- `--chunk-size, -c`: Posts per batch chunk (default: 1000)
- `--check-interval, -i`: Status check interval in seconds (default: 60)
- `--force, -f`: Overwrite existing files without prompting

This script:
- Processes posts in batches using OpenAI's batch API
- Creates embeddings for semantic content
- Outputs `{session}_embedded.jsonl`: Semantic content with embeddings

### 3. Dataset Analysis

Analyze posting patterns, engagement metrics, and temporal trends:

```bash
python analyze_dataset.py <session_name>
```

**Example:**
```bash
python analyze_dataset.py august
```

This generates:
- Posting frequency analysis (daily, hourly, by day of week)
- Engagement metrics (likes, reposts, replies, quotes)
- Author analysis (top authors by engagement)
- Interactive HTML dashboard with visualizations

### 4. Clustering Analysis

Perform semantic clustering on embedded posts:

```bash
python cluster_analysis.py <session_name>
```

**Example:**
```bash
python cluster_analysis.py august
```

**Options:**
- `--top-posts, -t`: Number of representative posts per cluster (default: 5)
- `--min-cluster-size`: Minimum posts per cluster (default: 50)
- `--min-samples`: Core point threshold for clustering (default: 10)

This produces:
- UMAP dimensionality reduction (high-dim → 50D → 2D)
- HDBSCAN clustering of posts
- Interactive visualization of clusters
- Representative posts for each cluster
- Clustered data output file

## File Structure

### Input Files
- `config.json`: Configuration for data collection and processing

### Pipeline Outputs
- `datasets/{run_name}_raw_data.jsonl`: Raw downloaded posts from Bluesky API
- `datasets/{run_name}_processed.jsonl`: Processed posts after filtering and enrichment
- `datasets/{run_name}_embedded.jsonl`: Semantic content with embeddings
- `datasets/{run_name}_with_clusters.jsonl`: Posts with cluster assignments (from clustering analysis)

### Analysis Outputs
- `{run_name}_dataset_analysis.html`: Interactive analysis dashboard
- `{run_name}_clustering_viz.html`: Interactive cluster visualization

## Pipeline Steps (Internal)

The main pipeline consists of 5 steps executed by `main_pipeline.py`:

1. **step1_download_data.py**: Downloads posts from Bluesky API based on config
2. **step2_filter_likes.py**: Filters posts meeting minimum engagement threshold
3. **step3_filter_content.py**: Removes spam and adult content
4. **step4_enrich_replies.py**: Adds parent/root post context to replies
5. **step5_extract_semantic.py**: Extracts semantic content for embedding

## Examples

### Complete workflow for a new analysis:

1. **Configure** your analysis in `config.json`
2. **Run the pipeline**:
   ```bash
   python main_pipeline.py
   ```
3. **Generate embeddings**:
   ```bash
   python embed_content.py august
   ```
4. **Analyze the dataset**:
   ```bash
   python analyze_dataset.py august
   ```
5. **Perform clustering**:
   ```bash
   python cluster_analysis.py august
   ```

### Quick dataset analysis only:

If you already have processed data:
```bash
python analyze_dataset.py august
```

### Clustering with custom parameters:

```bash
python cluster_analysis.py august --min-cluster-size 30 --top-posts 10
```

## Output Files

All output files are saved in the current directory or `datasets/` folder:

- **Analysis Dashboard**: `{run_name}_dataset_analysis.html` - Open in browser for interactive charts
- **Clustering Visualization**: `{run_name}_clustering_viz.html` - Interactive cluster exploration
- **Data Files**: Various `.jsonl` files in `datasets/` with processed and enriched data

## Tips

- Start with a short date range and fewer keywords to test the pipeline
- The embedding step can take time for large datasets due to API rate limits
- Interactive visualizations work best in modern web browsers
- Large datasets may require adjusting cluster parameters for meaningful results
- Check the HTML output files for detailed insights and exploration capabilities

## Troubleshooting

- **Missing API key**: Ensure `.env` file contains your OpenAI API key
- **No data found**: Check your keywords and date range in `config.json`
- **Embedding failures**: Verify API key and check for rate limits
- **Empty clusters**: Try reducing `min_cluster_size` parameter
- **Visualization issues**: Ensure plotly is installed and open HTML files in a modern browser
