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
8. **Screenshot** representative posts automatically
9. **Generate** comprehensive HTML reports with embedded images
10. **Visualize** results through interactive dashboards

## Requirements

- Python 3.8+
- **Bluesky account** with app password (for data collection)
- **Bluesky web password** (for screenshot generation)
- **OpenAI API key** (for embeddings)
- Required packages:
  - pandas
  - numpy
  - plotly
  - umap-learn
  - scikit-learn
  - openai
  - python-dotenv
  - playwright

## Setup

1. **Install dependencies** (recommended in a virtual environment):
   ```bash
   pip install pandas numpy plotly umap-learn scikit-learn openai python-dotenv playwright
   playwright install chromium
   ```

2. **Create a `.env` file** with your API keys and Bluesky credentials:
   ```
   # OpenAI API key (for embeddings)
   OPENAI_API_KEY=your_openai_api_key_here

   # Bluesky credentials (for data collection)
   BLUESKY_HANDLE=your_handle.bsky.social
   BLUESKY_APP_PASSWORD=your_app_password_here

   # Bluesky web password (for screenshot generation)
   BLUESKY_WEB_PASSWROD=your_regular_bluesky_password
   ```

   **To get a Bluesky app password:**
   - Go to Settings → Privacy and Security → App Passwords in your Bluesky app
   - Create a new app password for this project
   - Use your full handle (e.g., `username.bsky.social`) not just the username

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

### 4. Dimensionality Reduction

Reduce high-dimensional embeddings for clustering:

```bash
python reduce.py <session_name>
```

**Example:**
```bash
python reduce.py august
```

This applies UMAP to reduce embeddings from high-dimensional space to lower dimensions suitable for clustering.

### 5. Clustering Analysis

Perform semantic clustering on reduced embeddings:

```bash
python cluster.py <session_name>
```

**Example:**
```bash
python cluster.py august
```

This produces:
- Gaussian Mixture Model (GMM) clustering of posts
- Interactive visualization of clusters
- Clustered data output file with assignments

### 6. Post Screenshots

Generate screenshots of individual Bluesky posts:

```bash
python bluesky_screenshot.py <post_url>
```

**Example:**
```bash
python bluesky_screenshot.py "https://bsky.app/profile/username/post/postid"
```

**Options:**
- `-o, --output`: Custom output filename (optional)

This tool:
- Automatically logs into Bluesky using your credentials
- Navigates to the post and waits for content to load
- Takes a clean screenshot cropped to just the post content
- Saves as PNG file ready for sharing or embedding

### 7. Report Generation

Generate comprehensive HTML reports with embedded screenshots:

```bash
python generate_cluster_report.py <session_name>
```

**Example:**
```bash
python generate_cluster_report.py august
```

This creates a complete analysis report including:
- Executive summary with key statistics
- Detailed analysis of each relevant cluster
- Embedded screenshots of top 3 representative posts per cluster
- Clickable author profiles
- Temporal analysis charts
- Methodology and data pipeline documentation

## File Structure

### Input Files
- `config.json`: Configuration for data collection and processing

### Pipeline Outputs
- `datasets/{run_name}_raw_data.jsonl`: Raw downloaded posts from Bluesky API
- `datasets/{run_name}_processed.jsonl`: Processed posts after filtering and enrichment
- `datasets/{run_name}_embedded.jsonl`: Semantic content with embeddings
- `datasets/{run_name}_reduced.jsonl`: Dimensionality-reduced embeddings
- `datasets/{run_name}_clusters.jsonl`: Posts with cluster assignments
- `datasets/{run_name}_cluster_descriptions.jsonl`: AI-generated cluster descriptions

### Analysis Outputs
- `{run_name}_dataset_analysis.html`: Interactive analysis dashboard
- `{run_name}_clustering_viz.html`: Interactive cluster visualization
- `screenshots/`: Directory containing post screenshots (PNG files)
- `reports/{run_name}_cluster_report.html`: Comprehensive HTML report with embedded images

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
5. **Reduce dimensions**:
   ```bash
   python reduce.py august
   ```
6. **Perform clustering**:
   ```bash
   python cluster.py august
   ```
7. **Generate comprehensive report**:
   ```bash
   python generate_cluster_report.py august
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

- **Missing API keys**: Ensure `.env` file contains both `OPENAI_API_KEY` and Bluesky credentials (`BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`)
- **Authentication errors**: Verify your Bluesky handle includes the full domain (e.g., `username.bsky.social`) and your app password is correct
- **No data found**: Check your keywords and date range in `config.json`
- **Embedding failures**: Verify OpenAI API key and check for rate limits
- **Empty clusters**: Try reducing `min_cluster_size` parameter
- **Visualization issues**: Ensure plotly is installed and open HTML files in a modern browser
