#!/usr/bin/env python3
"""
Bluesky Cluster Analysis Report Generator
Usage: python generate_cluster_report.py <session_name>
Example: python generate_cluster_report.py august
"""

import json
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import base64
from io import BytesIO
import subprocess
import os
from collections import defaultdict, Counter
import numpy as np


class ClusterReportGenerator:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.datasets_dir = Path('datasets')
        self.screenshots_dir = Path('screenshots')
        self.screenshots_dir.mkdir(exist_ok=True)

        # File paths
        self.cluster_descriptions_file = self.datasets_dir / f"{session_name}_cluster_descriptions.jsonl"
        self.clusters_file = self.datasets_dir / f"{session_name}_clusters.jsonl"
        self.processed_file = self.datasets_dir / f"{session_name}_processed.jsonl"

        # Data containers
        self.cluster_descriptions = {}
        self.cluster_assignments = {}
        self.posts = {}
        self.relevant_cluster_ids = set()
        self.cluster_id_mapping = {}  # Maps original ID -> new sequential number

    def load_data(self):
        """Load all required data files"""
        print("📊 Loading data files...")

        # Load cluster descriptions
        print(f"  📖 Loading cluster descriptions from {self.cluster_descriptions_file}")
        if not self.cluster_descriptions_file.exists():
            raise FileNotFoundError(f"Cluster descriptions file not found: {self.cluster_descriptions_file}")

        with open(self.cluster_descriptions_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                cluster_id = data['cluster_id']
                self.cluster_descriptions[cluster_id] = data

                # Track relevant clusters
                if data.get('relevance') == 'relevant':
                    self.relevant_cluster_ids.add(cluster_id)

        print(f"    ✅ Loaded {len(self.cluster_descriptions)} cluster descriptions")
        print(f"    ✅ Found {len(self.relevant_cluster_ids)} relevant clusters")

        # Load cluster assignments
        print(f"  📖 Loading cluster assignments from {self.clusters_file}")
        if not self.clusters_file.exists():
            raise FileNotFoundError(f"Cluster assignments file not found: {self.clusters_file}")

        with open(self.clusters_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                uri = data['uri']
                self.cluster_assignments[uri] = data

        print(f"    ✅ Loaded {len(self.cluster_assignments)} cluster assignments")

        # Load processed posts
        print(f"  📖 Loading processed posts from {self.processed_file}")
        if not self.processed_file.exists():
            raise FileNotFoundError(f"Processed posts file not found: {self.processed_file}")

        with open(self.processed_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    uri = data['uri']
                    self.posts[uri] = data

                    if line_num % 50000 == 0:
                        print(f"    📊 Loaded {line_num:,} posts...")

                except json.JSONDecodeError:
                    continue

        print(f"    ✅ Loaded {len(self.posts):,} processed posts")

        # Create cluster ID mapping for sequential numbering
        self._create_cluster_id_mapping()

        return True

    def _create_cluster_id_mapping(self):
        """Create mapping from original cluster IDs to sequential numbers for relevant clusters only"""
        print("  📊 Creating cluster ID mapping for sequential numbering (relevant clusters only)...")

        # Get only relevant clusters sorted by post count (descending)
        relevant_clusters_by_size = []
        for cluster_id, cluster_desc in self.cluster_descriptions.items():
            is_relevant = cluster_desc.get('relevance') == 'relevant'
            if is_relevant:
                post_count = cluster_desc.get('post_count', 0)
                relevant_clusters_by_size.append((post_count, cluster_id))

        # Sort by post count (descending)
        relevant_clusters_by_size.sort(key=lambda x: x[0], reverse=True)

        # Create mapping for relevant clusters only: original_id -> display_number (1, 2, 3...)
        for display_number, (post_count, original_id) in enumerate(relevant_clusters_by_size, 1):
            self.cluster_id_mapping[original_id] = display_number

        # Irrelevant clusters keep their original IDs (no mapping)

    def get_display_cluster_id(self, original_id: int) -> int:
        """Get the display cluster ID (sequential number) for an original cluster ID"""
        return self.cluster_id_mapping.get(original_id, original_id)

    def get_posts_for_cluster(self, cluster_id: int):
        """Get all posts assigned to a specific cluster"""
        posts_in_cluster = []

        for uri, assignment in self.cluster_assignments.items():
            if assignment['cluster_id'] == cluster_id and uri in self.posts:
                post_data = self.posts[uri].copy()
                post_data.update(assignment)  # Add cluster info
                posts_in_cluster.append(post_data)

        return posts_in_cluster


    def calculate_post_score(self, post):
        """Calculate post score: likeCount * probability^2"""
        like_count = post.get('likeCount', 0)
        probability = post.get('cluster_probability', 0)
        return like_count * (probability ** 2)

    def get_subcluster_labels_from_data(self, posts_with_embeddings):
        """Get existing subcluster_id values from the cluster data"""
        subcluster_labels = []

        for post in posts_with_embeddings:
            uri = post.get('uri')
            if uri in self.cluster_assignments:
                # Get subcluster_id from cluster assignments, default to 0 if not found
                subcluster_id = self.cluster_assignments[uri].get('subcluster_id', 0)
                subcluster_labels.append(subcluster_id)
            else:
                subcluster_labels.append(0)

        return subcluster_labels

    def select_representative_posts_from_subclusters(self, posts_with_embeddings, sub_cluster_labels):
        """Select the top post by likes from each sub-cluster"""
        # Group posts by sub-cluster
        subcluster_posts = {}
        for post, sub_label in zip(posts_with_embeddings, sub_cluster_labels):
            if sub_label not in subcluster_posts:
                subcluster_posts[sub_label] = []
            subcluster_posts[sub_label].append(post)

        representative_posts = []

        # For each sub-cluster, get the post with the highest like count
        for sub_label in sorted(subcluster_posts.keys()):
            subcluster_post_list = subcluster_posts[sub_label]

            # Sort by like count descending and take the top one
            best_post = max(subcluster_post_list, key=lambda p: p.get('likeCount', 0))
            representative_posts.append(best_post)

        return representative_posts

    def get_top_posts_for_cluster(self, cluster_id: int, num_posts: int = 3):
        """Get top representative posts with unique authors by scoring"""
        cluster_posts = self.get_posts_for_cluster(cluster_id)

        # Calculate scores for all posts
        scored_posts = []
        for post in cluster_posts:
            score = self.calculate_post_score(post)
            scored_posts.append((score, post))

        # Sort by score descending
        scored_posts.sort(key=lambda x: x[0], reverse=True)

        # Select top posts ensuring unique authors
        selected_posts = []
        used_authors = set()

        for score, post in scored_posts:
            author_handle = post.get('author', {}).get('handle', 'unknown')

            # Skip if we already have a post from this author
            if author_handle in used_authors:
                continue

            selected_posts.append(post)
            used_authors.add(author_handle)

            # Stop when we have enough posts
            if len(selected_posts) >= num_posts:
                break

        print(f"    🎯 Selected {len(selected_posts)} posts with unique authors for cluster {cluster_id}")
        return selected_posts

    def url_from_post(self, post):
        """Convert post data to Bluesky URL"""
        author_handle = post.get('author', {}).get('handle', '')
        uri = post.get('uri', '')

        # Extract post ID from AT URI
        # Format: at://did:plc:xxx/app.bsky.feed.post/POST_ID
        if '/app.bsky.feed.post/' in uri:
            post_id = uri.split('/app.bsky.feed.post/')[-1]
            return f"https://bsky.app/profile/{author_handle}/post/{post_id}"

        return None

    def generate_screenshots_for_cluster(self, cluster_id: int):
        """Generate screenshots for top 3 posts in a cluster"""
        print(f"  📸 Getting screenshots for cluster {cluster_id}...")

        top_posts = self.get_top_posts_for_cluster(cluster_id, 3)
        screenshot_info = []
        urls_to_screenshot = []

        for i, post in enumerate(top_posts):
            url = self.url_from_post(post)
            if not url:
                print(f"    ⚠️  Could not generate URL for post {i+1}")
                continue

            author_handle = post.get('author', {}).get('handle', 'unknown')
            uri = post.get('uri', '')
            post_id = uri.split('/app.bsky.feed.post/')[-1] if '/app.bsky.feed.post/' in uri else 'unknown'

            screenshot_filename = f"bluesky_post_{author_handle}_{post_id}.png"
            screenshot_path = self.screenshots_dir / screenshot_filename

            # Check if screenshot already exists
            if screenshot_path.exists():
                screenshot_info.append({
                    'post': post,
                    'url': url,
                    'screenshot_path': screenshot_path,
                    'screenshot_filename': screenshot_filename,
                    'score': self.calculate_post_score(post)
                })
            else:
                urls_to_screenshot.append(url)
                screenshot_info.append({
                    'post': post,
                    'url': url,
                    'screenshot_path': screenshot_path,
                    'screenshot_filename': screenshot_filename,
                    'score': self.calculate_post_score(post)
                })

        # Generate missing screenshots if any
        if urls_to_screenshot:
            print(f"    📸 Generating {len(urls_to_screenshot)} screenshots...")
            print(f"    🔗 URLs: {urls_to_screenshot[:3]}...")  # Show first 3 URLs for debugging
            self.generate_screenshots_batch(urls_to_screenshot)

        return screenshot_info

    def generate_screenshots_batch(self, urls):
        """Generate screenshots for a batch of URLs using our screenshot script"""
        if not urls:
            return True

        success_count = 0
        total_urls = len(urls)

        for i, url in enumerate(urls, 1):
            print(f"    📸 Generating screenshot {i}/{total_urls}: {url}")

            try:
                # Extract filename info from URL for custom output path
                # URL format: https://bsky.app/profile/username/post/postid
                url_parts = url.split('/')
                if len(url_parts) >= 6:
                    username = url_parts[4]  # profile/username
                    post_id = url_parts[6]   # post/postid
                    output_filename = f"bluesky_post_{username}_{post_id}.png"
                    output_path = self.screenshots_dir / output_filename
                else:
                    # Fallback filename
                    output_filename = f"bluesky_post_{i}.png"
                    output_path = self.screenshots_dir / output_filename

                # Skip if screenshot already exists
                if output_path.exists():
                    print(f"    ✅ Screenshot already exists: {output_filename}")
                    success_count += 1
                    continue

                # Build command to run our screenshot script for single URL
                cmd = [
                    sys.executable, 'bluesky_screenshot.py',
                    '-o', str(output_path),
                    url
                ]

                # Run the screenshot script with longer timeout
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout per screenshot
                )

                if result.returncode == 0:
                    print(f"    ✅ Screenshot saved: {output_filename}")
                    success_count += 1
                else:
                    print(f"    ❌ Screenshot failed for {url}:")
                    print(f"    stdout: {result.stdout}")
                    print(f"    stderr: {result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"    ❌ Screenshot timed out after 2 minutes: {url}")
            except Exception as e:
                print(f"    ❌ Error generating screenshot for {url}: {e}")

        print(f"    📊 Screenshot generation complete: {success_count}/{total_urls} successful")
        return success_count > 0  # Return True if at least one screenshot was generated

    def screenshot_to_base64(self, screenshot_path):
        """Convert screenshot to base64 for embedding in HTML"""
        try:
            with open(screenshot_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                return f"data:image/png;base64,{base64_data}"
        except Exception as e:
            print(f"    ❌ Error converting screenshot to base64: {e}")
            return None

    def test_screenshot_functionality(self):
        """Test screenshot generation for one cluster"""
        print("🧪 Testing screenshot functionality...")

        # Get the first relevant cluster
        if not self.relevant_cluster_ids:
            print("  ❌ No relevant clusters found")
            return False

        test_cluster_id = next(iter(self.relevant_cluster_ids))
        print(f"  🎯 Testing with cluster {test_cluster_id}")

        # Test getting top posts
        top_posts = self.get_top_posts_for_cluster(test_cluster_id, 3)
        print(f"  📊 Found {len(top_posts)} top posts for cluster {test_cluster_id}")

        for i, post in enumerate(top_posts, 1):
            score = self.calculate_post_score(post)
            author = post.get('author', {}).get('displayName', 'Unknown')
            likes = post.get('likeCount', 0)
            prob = post.get('cluster_probability', 0)
            print(f"    {i}. {author}: {likes} likes × {prob:.3f}² = {score:.1f} score")

        # Test screenshot generation
        screenshot_info = self.generate_screenshots_for_cluster(test_cluster_id)
        print(f"  📸 Generated screenshot info for {len(screenshot_info)} posts")

        return True

    def generate_report(self):
        """Generate the complete HTML report"""
        print("📝 Generating HTML report...")

        # Generate sections in desired order
        html_content = self._generate_html_header()
        html_content += self._generate_title_section()  # Title at top
        html_content += self._generate_methodology_section()  # Methodology back at top
        html_content += self._generate_executive_summary()
        html_content += self._generate_cluster_summary_table()
        html_content += self._generate_individual_cluster_sections()
        html_content += self._generate_irrelevant_clusters_section()
        html_content += self._generate_temporal_analysis()  # Moved to end
        html_content += self._generate_html_footer()

        # Save report
        output_file = f"reports/{self.session_name}_cluster_report.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Report generated: {output_file}")
        return output_file

    def _generate_html_header(self):
        """Generate HTML header with embedded CSS"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bluesky Cluster Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1 { color: #1a472a; border-bottom: 3px solid #4a9eff; padding-bottom: 10px; }
        h2 { color: #2c5aa0; margin-top: 40px; }
        h3 { color: #4a9eff; }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #4a9eff;
        }
        .stat-number { font-size: 2em; font-weight: bold; color: #1a472a; }
        .stat-label { color: #666; font-size: 0.9em; }
        .cluster-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .screenshots {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .screenshot {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        .screenshot img {
            width: 100%;
            height: auto;
        }
        .metadata { background: #f0f8ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; }
        .chart { text-align: center; margin: 30px 0; }
        .keywords {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0;
        }
        .keyword {
            background: #e3f2fd;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            border: 1px solid #4a9eff;
        }
        @media print {
            .cluster-card { page-break-inside: avoid; }
            .screenshots { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
"""

    def _generate_html_footer(self):
        """Generate HTML footer"""
        return """
</body>
</html>
"""

    def _generate_title_section(self):
        """Generate the main title section"""
        return f"""
<h1>Bluesky Cluster Analysis Report - {self.session_name.title()}</h1>
"""

    def _generate_methodology_section(self):
        """Generate methodology and metadata section"""
        # Load config file to get keywords and date range
        config_file = Path('config.json')
        config_data = {}

        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)

        keywords = config_data.get('keywords', ['Not available'])
        date_range = config_data.get('date_range', {})
        min_likes = config_data.get('filters', {}).get('min_likes', 'Unknown')

        # Count original raw posts
        raw_data_file = self.datasets_dir / f"{self.session_name}_raw_data.jsonl"
        raw_posts_count = 0
        if raw_data_file.exists():
            with open(raw_data_file, 'r') as f:
                raw_posts_count = sum(1 for line in f if line.strip())

        # Format keywords nicely
        keywords_html = ""
        if isinstance(keywords, list):
            keywords_html = '<div class="keywords">'
            for keyword in keywords:  # Show all keywords
                keywords_html += f'<span class="keyword">{keyword}</span>'
            keywords_html += '</div>'

        date_range_str = "Unknown"
        if date_range:
            start = date_range.get('start_date', 'Unknown')
            end = date_range.get('end_date', 'Unknown')
            date_range_str = f"{start} to {end}"

        return f"""
<div class="metadata">
    <h2>📋 Methodology & Data Collection</h2>
    <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Session:</strong> {self.session_name}</p>
    <p><strong>Date Range:</strong> {date_range_str}</p>

    <h3>Search Keywords ({len(keywords)} total):</h3>
    {keywords_html}

    <h3>Data Processing Pipeline:</h3>
    <ol>
        <li><strong>Data Collection:</strong> {raw_posts_count:,} posts downloaded via Bluesky API using above keywords</li>
        <li><strong>Like Threshold:</strong> Filtered to posts with ≥{min_likes} likes</li>
        <li><strong>Content Filtering:</strong> Removed posts labeled as:
            <ul>
                <li>Spam content</li>
                <li>Adult content (porn, sexual, nudity, sexual-figurative, graphic-media)</li>
            </ul>
        </li>
        <li><strong>Reply Enrichment:</strong> Added parent/root post context for replies</li>
        <li><strong>Semantic Extraction:</strong> Generated semantic content for embedding</li>
        <li><strong>Clustering:</strong> Applied UMAP + Gaussian Mixture Model (GMM) clustering algorithm</li>
        <li><strong>AI Classification:</strong> Cluster descriptions and relevance labels generated by GPT-5 mini</li>
    </ol>

    <h3>Final Dataset Statistics:</h3>
    <ul>
        <li><strong>Total Posts Analyzed:</strong> {len(self.posts):,}</li>
        <li><strong>Total Clusters Identified:</strong> {len(self.cluster_descriptions)}</li>
        <li><strong>Relevant Clusters:</strong> {len(self.relevant_cluster_ids)}</li>
        <li><strong>Irrelevant Clusters:</strong> {len(self.cluster_descriptions) - len(self.relevant_cluster_ids)}</li>
    </ul>
</div>
"""

    def _generate_executive_summary(self):
        """Generate executive summary section"""
        print("  📊 Generating executive summary...")

        # Calculate overall stats for relevant clusters only
        relevant_posts = []
        total_likes = 0
        total_reposts = 0
        total_replies = 0
        author_stats = defaultdict(lambda: {'posts': 0, 'likes': 0, 'handle': '', 'display_name': ''})

        for cluster_id in self.relevant_cluster_ids:
            cluster_posts = self.get_posts_for_cluster(cluster_id)
            relevant_posts.extend(cluster_posts)

            for post in cluster_posts:
                # Overall engagement stats
                total_likes += post.get('likeCount', 0)
                total_reposts += post.get('repostCount', 0)
                total_replies += post.get('replyCount', 0)

                # Author stats
                author = post.get('author', {})
                author_handle = author.get('handle', 'unknown')
                author_stats[author_handle]['posts'] += 1
                author_stats[author_handle]['likes'] += post.get('likeCount', 0)
                author_stats[author_handle]['handle'] = author_handle
                author_stats[author_handle]['display_name'] = author.get('displayName', author_handle)

        # Top 20 authors by total likes
        top_authors = sorted(author_stats.items(), key=lambda x: x[1]['likes'], reverse=True)[:20]

        # Calculate averages
        avg_likes = total_likes / len(relevant_posts) if relevant_posts else 0
        avg_reposts = total_reposts / len(relevant_posts) if relevant_posts else 0
        avg_replies = total_replies / len(relevant_posts) if relevant_posts else 0

        # Generate 2D cluster visualization
        cluster_viz_html = self._generate_2d_cluster_visualization()

        html = f"""
<h2>📊 Executive Summary</h2>

<div class="summary-stats">
    <div class="stat-card">
        <div class="stat-number">{len(relevant_posts):,}</div>
        <div class="stat-label">Relevant Posts</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{len(self.relevant_cluster_ids)}</div>
        <div class="stat-label">Relevant Clusters</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{total_likes:,}</div>
        <div class="stat-label">Total Likes</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{avg_likes:.1f}</div>
        <div class="stat-label">Avg Likes/Post</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{len(author_stats):,}</div>
        <div class="stat-label">Unique Authors</div>
    </div>
</div>

{cluster_viz_html}

<h3>🏆 Top 20 Authors (by total likes in relevant clusters)</h3>
<table>
    <thead>
        <tr>
            <th>Rank</th>
            <th>Author</th>
            <th>Posts</th>
            <th>Total Likes</th>
            <th>Avg Likes/Post</th>
        </tr>
    </thead>
    <tbody>
"""

        for i, (handle, stats) in enumerate(top_authors, 1):
            avg_likes_per_post = stats['likes'] / stats['posts'] if stats['posts'] > 0 else 0
            display_name = stats['display_name'] if stats['display_name'] != handle else handle

            html += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{display_name}</strong><br><small><a href="https://bsky.app/profile/{handle}" target="_blank">@{handle}</a></small></td>
            <td>{stats['posts']}</td>
            <td>{stats['likes']:,}</td>
            <td>{avg_likes_per_post:.1f}</td>
        </tr>
"""

        html += """
    </tbody>
</table>
"""

        return html

    def _generate_2d_cluster_visualization(self):
        """Generate 2D cluster visualization for relevant clusters only"""
        print("  🎨 Generating 2D cluster visualization...")

        # Collect 2D positions and cluster labels for relevant clusters only
        plot_data = []

        for cluster_id in self.relevant_cluster_ids:
            cluster_posts = self.get_posts_for_cluster(cluster_id)
            display_id = self.get_display_cluster_id(cluster_id)
            cluster_desc = self.cluster_descriptions[cluster_id]
            cluster_name = f"Cluster {display_id}"

            for post in cluster_posts:
                # Filter for posts with at least 100 likes for better visualization performance
                likes = post.get('likeCount', 0)
                if likes < 100:
                    continue

                uri = post.get('uri')
                if uri in self.cluster_assignments:
                    cluster_assignment = self.cluster_assignments[uri]
                    if 'umap_2d_coords' in cluster_assignment:
                        coords = cluster_assignment['umap_2d_coords']
                        plot_data.append({
                            'x': coords['x'],
                            'y': coords['y'],
                            'cluster_id': cluster_id,
                            'display_id': display_id,
                            'cluster_name': cluster_name,
                            'likes': likes,
                            'author': post.get('author', {}).get('displayName', 'Unknown')
                        })

        if not plot_data:
            return """
<h3>📍 2D Cluster Visualization</h3>
<p>No 2D coordinate data available for visualization.</p>
"""

        # Prepare data for Chart.js scatter plot
        datasets = []

        # Create color palette for clusters
        colors = [
            '#4a9eff', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4',
            '#ffeaa7', '#dda0dd', '#98d8c8', '#ffaaa5', '#a8e6cf',
            '#ff8b94', '#88d8b0', '#77c4d8', '#ffcc5c', '#b4a7d6'
        ]

        # Group data by cluster
        cluster_data = {}
        for point in plot_data:
            cluster_name = point['cluster_name']
            if cluster_name not in cluster_data:
                cluster_data[cluster_name] = {'x': [], 'y': [], 'labels': []}

            cluster_data[cluster_name]['x'].append(point['x'])
            cluster_data[cluster_name]['y'].append(point['y'])
            cluster_data[cluster_name]['labels'].append(f"{point['author']} ({point['likes']} likes)")

        # Create dataset for each cluster with proper JSON formatting
        datasets_json = []
        tooltip_data = {}

        # Sort by display ID number, not cluster name string
        sorted_clusters = sorted(cluster_data.items(), key=lambda x: int(x[0].split()[1]))
        for i, (cluster_name, data) in enumerate(sorted_clusters):
            color = colors[i % len(colors)]

            # Create properly formatted data points
            data_points = []
            for j, (x, y) in enumerate(zip(data['x'], data['y'])):
                data_points.append(f"{{x: {x}, y: {y}}}")

            datasets_json.append(f"""{{
                label: "{cluster_name}",
                data: [{', '.join(data_points)}],
                backgroundColor: "{color}",
                borderColor: "{color}",
                pointRadius: 3,
                pointHoverRadius: 5
            }}""")

            # Store tooltip data separately
            tooltip_data[cluster_name] = data['labels']

        chart_id = "cluster-2d-viz"

        # Debug info
        print(f"    📊 Plotting {len(plot_data)} posts across {len(cluster_data)} clusters")
        for cluster_name, data in cluster_data.items():
            print(f"      {cluster_name}: {len(data['x'])} points")

        html = f"""
<h3>📍 2D Cluster Visualization</h3>
<p>Interactive scatter plot showing the spatial distribution of posts from relevant clusters in 2D embedding space. Each point represents a post with ≥100 likes, colored by cluster assignment.</p>
<p><small>Plotting {len(plot_data):,} high-engagement posts (≥100 likes) across {len(cluster_data)} clusters</small></p>

<div style="margin: 30px 0;">
    <canvas id="{chart_id}" width="800" height="600"></canvas>
</div>

<script>
console.log('🔧 Starting 2D cluster visualization...');

// Debug: Check if Chart.js is available
if (typeof Chart === 'undefined') {{
    console.error('❌ Chart.js not loaded!');
    document.getElementById('{chart_id}').innerHTML = '<p style="color: red;">Error: Chart.js not loaded</p>';
}} else {{
    console.log('✅ Chart.js is available');
}}

// Debug: Check canvas element
const canvasElement = document.getElementById('{chart_id}');
if (!canvasElement) {{
    console.error('❌ Canvas element not found!');
}} else {{
    console.log('✅ Canvas element found');
}}

const ctx2d = canvasElement.getContext('2d');
const tooltipData = {tooltip_data};

console.log('📊 Tooltip data:', Object.keys(tooltipData).length, 'clusters');

try {{
    const clusterChart = new Chart(ctx2d, {{
        type: 'scatter',
        data: {{
            datasets: [{', '.join(datasets_json)}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{
                    display: true,
                    text: '2D Cluster Distribution (Relevant Clusters Only)'
                }},
                legend: {{
                    display: true,
                    position: 'bottom'
                }},
                tooltip: {{
                    callbacks: {{
                        title: function(context) {{
                            return context[0].dataset.label;
                        }},
                        label: function(context) {{
                            const pointIndex = context.dataIndex;
                            const clusterName = context.dataset.label;
                            return tooltipData[clusterName] ? tooltipData[clusterName][pointIndex] : 'No data';
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{
                    type: 'linear',
                    position: 'bottom',
                    title: {{
                        display: true,
                        text: 'UMAP Dimension 1'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'UMAP Dimension 2'
                    }}
                }}
            }},
            interaction: {{
                intersect: false
            }}
        }}
    }});

    console.log('✅ Chart created successfully');

}} catch (error) {{
    console.error('❌ Error creating chart:', error);
    document.getElementById('{chart_id}').innerHTML = '<p style="color: red;">Error creating chart: ' + error.message + '</p>';
}}
</script>
"""

        return html

    def _generate_temporal_analysis(self):
        """Generate temporal analysis with posts/likes per day chart"""
        print("  📈 Generating temporal analysis...")

        # Collect daily stats (only relevant clusters)
        daily_stats = defaultdict(lambda: {'posts': 0, 'likes': 0})

        for cluster_id in self.relevant_cluster_ids:
            cluster_posts = self.get_posts_for_cluster(cluster_id)

            for post in cluster_posts:
                # Parse the date from the post
                created_at = post.get('record', {}).get('createdAt', '')
                if created_at:
                    try:
                        # Extract date (YYYY-MM-DD) from ISO timestamp
                        date_str = created_at[:10]  # Get YYYY-MM-DD part
                        daily_stats[date_str]['posts'] += 1
                        daily_stats[date_str]['likes'] += post.get('likeCount', 0)
                    except:
                        continue

        # Sort by date and prepare data for chart
        sorted_dates = sorted(daily_stats.keys())

        if not sorted_dates:
            return """
<div style="margin: 60px 0 40px 0; border-top: 3px solid #999; padding-top: 40px;">
<h2>📈 Temporal Analysis</h2>
<p>No temporal data available for analysis.</p>
</div>
"""

        # Prepare chart data
        dates = []
        posts_per_day = []
        likes_per_day = []

        for date in sorted_dates:
            dates.append(date)
            posts_per_day.append(daily_stats[date]['posts'])
            likes_per_day.append(daily_stats[date]['likes'])

        # Create chart using Chart.js
        chart_id = "temporal-chart"

        html = f"""
<div style="margin: 60px 0 40px 0; border-top: 3px solid #999; padding-top: 40px;">
<h2>📈 Temporal Analysis</h2>
<p>Posts and likes per day for relevant clusters over the analysis period.</p>

<div style="margin: 30px 0;">
    <canvas id="{chart_id}" width="800" height="400"></canvas>
</div>

<script>
const ctx = document.getElementById('{chart_id}').getContext('2d');
const chart = new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: {dates},
        datasets: [
            {{
                label: 'Posts per Day',
                data: {posts_per_day},
                borderColor: '#4a9eff',
                backgroundColor: 'rgba(74, 158, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                yAxisID: 'y'
            }},
            {{
                label: 'Likes per Day',
                data: {likes_per_day},
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                borderWidth: 2,
                fill: true,
                yAxisID: 'y1'
            }}
        ]
    }},
    options: {{
        responsive: true,
        interaction: {{
            mode: 'index',
            intersect: false,
        }},
        plugins: {{
            title: {{
                display: true,
                text: 'Temporal Activity Pattern'
            }},
            legend: {{
                display: true
            }}
        }},
        scales: {{
            x: {{
                display: true,
                title: {{
                    display: true,
                    text: 'Date'
                }}
            }},
            y: {{
                type: 'linear',
                display: true,
                position: 'left',
                beginAtZero: true,
                title: {{
                    display: true,
                    text: 'Posts'
                }}
            }},
            y1: {{
                type: 'linear',
                display: true,
                position: 'right',
                beginAtZero: true,
                title: {{
                    display: true,
                    text: 'Likes'
                }},
                grid: {{
                    drawOnChartArea: false,
                }},
            }}
        }}
    }}
}});
</script>

</div>
"""

        return html

    def _generate_cluster_summary_table(self):
        """Generate cluster summary table showing all clusters"""
        print("  📋 Generating cluster summary table...")

        # Get relevant clusters only, sorted by post count
        relevant_clusters = []
        for cluster_id, cluster_desc in self.cluster_descriptions.items():
            is_relevant = cluster_desc.get('relevance') == 'relevant'

            # Only include relevant clusters
            if is_relevant:
                post_count = cluster_desc.get('post_count', 0)

                # Calculate total likes for this cluster
                cluster_posts = self.get_posts_for_cluster(cluster_id)
                total_likes = sum(post.get('likeCount', 0) for post in cluster_posts)

                relevant_clusters.append((post_count, cluster_id, cluster_desc, total_likes))

        # Sort by post count (descending)
        relevant_clusters.sort(key=lambda x: x[0], reverse=True)

        html = f"""
<h2>📋 Relevant Clusters Summary</h2>
<p>Overview of the {len(relevant_clusters)} relevant clusters identified in the analysis, ordered by post count.</p>

<div class="cluster-summary-table">
    <table>
        <thead>
            <tr>
                <th>Cluster ID</th>
                <th>Posts</th>
                <th>Total Likes</th>
                <th>Avg Likes/Post</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
"""

        for post_count, cluster_id, cluster_desc, total_likes in relevant_clusters:
            avg_likes = total_likes / post_count if post_count > 0 else 0
            description = cluster_desc.get('theme', 'No description available')
            display_id = self.get_display_cluster_id(cluster_id)

            html += f"""
            <tr>
                <td><strong>Cluster {display_id}</strong></td>
                <td>{post_count:,}</td>
                <td>{total_likes:,}</td>
                <td>{avg_likes:.1f}</td>
                <td>{description}</td>
            </tr>
"""

        html += """
        </tbody>
    </table>
</div>
"""

        return html

    def _generate_individual_cluster_sections(self):
        """Generate individual cluster sections with screenshots"""
        print("  🎯 Generating individual cluster sections...")

        html = """
<div style="margin: 60px 0 40px 0; border-top: 3px solid #4a9eff; padding-top: 40px;">
<h2>🎯 Individual Cluster Analysis</h2>
<p>Detailed analysis of each relevant cluster, ordered by number of posts.</p>
</div>
"""

        # Sort relevant clusters by post count (descending)
        relevant_clusters_info = []
        for cluster_id in self.relevant_cluster_ids:
            cluster_desc = self.cluster_descriptions[cluster_id]
            post_count = cluster_desc.get('post_count', 0)
            relevant_clusters_info.append((post_count, cluster_id, cluster_desc))

        relevant_clusters_info.sort(key=lambda x: x[0], reverse=True)

        # Generate section for each cluster
        for i, (post_count, cluster_id, cluster_desc) in enumerate(relevant_clusters_info, 1):
            display_id = self.get_display_cluster_id(cluster_id)
            print(f"    📊 Processing cluster {cluster_id} -> {display_id} ({post_count:,} posts)...")

            # Get cluster posts and calculate stats
            cluster_posts = self.get_posts_for_cluster(cluster_id)
            total_likes = sum(post.get('likeCount', 0) for post in cluster_posts)
            total_reposts = sum(post.get('repostCount', 0) for post in cluster_posts)
            total_replies = sum(post.get('replyCount', 0) for post in cluster_posts)
            avg_likes = total_likes / len(cluster_posts) if cluster_posts else 0

            # Get top authors for this cluster
            author_stats = defaultdict(lambda: {'posts': 0, 'likes': 0, 'handle': '', 'display_name': ''})
            for post in cluster_posts:
                author = post.get('author', {})
                author_handle = author.get('handle', 'unknown')
                author_stats[author_handle]['posts'] += 1
                author_stats[author_handle]['likes'] += post.get('likeCount', 0)
                author_stats[author_handle]['handle'] = author_handle
                author_stats[author_handle]['display_name'] = author.get('displayName', author_handle)

            top_authors = sorted(author_stats.items(), key=lambda x: x[1]['likes'], reverse=True)[:5]

            # Generate screenshots for this cluster
            screenshot_info = self.generate_screenshots_for_cluster(cluster_id)

            # Build HTML for this cluster
            html += f"""
<div class="cluster-card">
    <h3>Cluster {display_id}</h3>

    <div class="summary-stats">
        <div class="stat-card">
            <div class="stat-number">{post_count:,}</div>
            <div class="stat-label">Posts</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_likes:,}</div>
            <div class="stat-label">Total Likes</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{avg_likes:.1f}</div>
            <div class="stat-label">Avg Likes/Post</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(author_stats):,}</div>
            <div class="stat-label">Unique Authors</div>
        </div>
    </div>

    <h4>📝 Description</h4>
    <p>{cluster_desc.get('theme', 'No description available')}</p>

    <h4>🏷️ Keywords</h4>
    <div class="keywords">
"""

            # Add keywords
            keywords = cluster_desc.get('keywords', [])
            for keyword in keywords:
                html += f'<span class="keyword">{keyword}</span>'

            html += """
    </div>

    <h4>📸 Top 3 Representative Posts</h4>
    <div class="screenshots">
"""

            # Add screenshots
            for j, screenshot in enumerate(screenshot_info, 1):
                post = screenshot['post']
                author = post.get('author', {})
                author_name = author.get('displayName', author.get('handle', 'Unknown'))
                likes = post.get('likeCount', 0)
                score = screenshot['score']

                # Convert screenshot to base64 for embedding
                base64_image = self.screenshot_to_base64(screenshot['screenshot_path'])

                if base64_image:
                    html += f"""
        <div class="screenshot">
            <img src="{base64_image}" alt="Post by {author_name}">
            <div style="padding: 10px; background: #f8f9fa;">
                <strong>#{j}: {author_name}</strong><br>
                <small><a href="{screenshot['url']}" target="_blank">View original</a></small>
            </div>
        </div>
"""
                else:
                    html += f"""
        <div class="screenshot">
            <p>Screenshot not available for post by {author_name}</p>
        </div>
"""

            html += """
    </div>

    <h4>🏆 Top 5 Authors in this Cluster</h4>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Author</th>
                <th>Posts</th>
                <th>Total Likes</th>
                <th>Avg Likes/Post</th>
            </tr>
        </thead>
        <tbody>
"""

            # Add top authors for this cluster
            for rank, (handle, stats) in enumerate(top_authors, 1):
                avg_likes_per_post = stats['likes'] / stats['posts'] if stats['posts'] > 0 else 0
                display_name = stats['display_name'] if stats['display_name'] != handle else handle

                html += f"""
            <tr>
                <td>{rank}</td>
                <td><strong>{display_name}</strong><br><small><a href="https://bsky.app/profile/{handle}" target="_blank">@{handle}</a></small></td>
                <td>{stats['posts']}</td>
                <td>{stats['likes']:,}</td>
                <td>{avg_likes_per_post:.1f}</td>
            </tr>
"""

            html += """
        </tbody>
    </table>
</div>
"""

        return html

    def _generate_irrelevant_clusters_section(self):
        """Generate irrelevant clusters section"""
        print("  📊 Generating irrelevant clusters section...")

        # Get irrelevant clusters
        irrelevant_clusters = []
        for cluster_id, cluster_desc in self.cluster_descriptions.items():
            if cluster_desc.get('relevance') != 'relevant':
                post_count = cluster_desc.get('post_count', 0)
                irrelevant_clusters.append((post_count, cluster_id, cluster_desc))

        # Sort by post count (descending)
        irrelevant_clusters.sort(key=lambda x: x[0], reverse=True)

        html = f"""
<div style="margin: 60px 0 40px 0; border-top: 3px solid #999; padding-top: 40px;">
<h2>🗂️ Irrelevant Clusters</h2>
<p>The following {len(irrelevant_clusters)} clusters were identified but deemed irrelevant to the AI/technology theme, ordered by post count.</p>
</div>

<div class="cluster-summary-table">
    <table>
        <thead>
            <tr>
                <th>Cluster ID</th>
                <th>Posts</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
"""

        for i, (post_count, cluster_id, cluster_desc) in enumerate(irrelevant_clusters):
            description = cluster_desc.get('theme', 'No description available')

            html += f"""
            <tr>
                <td>Cluster {i}</td>
                <td>{post_count:,}</td>
                <td>{description}</td>
            </tr>
"""

        html += """
        </tbody>
    </table>
</div>
"""

        return html


def main():
    parser = argparse.ArgumentParser(description='Generate Bluesky cluster analysis report')
    parser.add_argument('session_name', help='Name of the session (e.g., august)')

    args = parser.parse_args()

    print(f"🚀 Generating cluster report for session: {args.session_name}")

    try:
        generator = ClusterReportGenerator(args.session_name)
        generator.load_data()
        output_file = generator.generate_report()

        print(f"🎉 Success! Report saved to: {output_file}")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())