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
import re
from urllib.parse import urlparse


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

        # Search config
        self.search_config = self.load_search_config()

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

    def load_search_config(self):
        """Load search configuration from search_config.json"""
        config_file = Path('search_config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load search config: {e}")
                return {"search_groups": []}
        else:
            print("⚠️  Warning: search_config.json not found - special searches will be skipped")
            return {"search_groups": []}

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

    def extract_links_from_post(self, post):
        """Extract all external links from a post"""
        links = set()

        # Extract from text using regex
        text = post.get('record', {}).get('text', '')
        if text:
            url_pattern = r'https?://[^\s<>"\'{}|\\^`\[\]]+[^\s<>"\'{}|\\^`\[\].,;:!?]'
            text_urls = re.findall(url_pattern, text)
            for url in text_urls:
                # Filter out Bluesky internal URLs and media URLs
                if not self._is_internal_url(url):
                    links.add(url)

        # Extract from facets (formatted links)
        facets = post.get('record', {}).get('facets', [])
        for facet in facets:
            for feature in facet.get('features', []):
                if feature.get('$type') == 'app.bsky.richtext.facet#link':
                    url = feature.get('uri', '')
                    if url and not self._is_internal_url(url):
                        links.add(url)

        # Extract from external embeds
        embed = post.get('record', {}).get('embed', {})
        if embed.get('$type') == 'app.bsky.embed.external':
            external_url = embed.get('external', {}).get('uri', '')
            if external_url and not self._is_internal_url(external_url):
                links.add(external_url)

        # Extract from embed view (for rendered embeds)
        embed_view = post.get('embed', {})
        if embed_view.get('$type') == 'app.bsky.embed.external#view':
            external_url = embed_view.get('external', {}).get('uri', '')
            if external_url and not self._is_internal_url(external_url):
                links.add(external_url)

        return list(links)

    def _is_internal_url(self, url):
        """Check if URL is Bluesky internal or media URL that should be filtered out"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Filter out Bluesky domains and CDN URLs
            internal_domains = {
                'bsky.app', 'cdn.bsky.app', 'bsky.social',
                'at.bsky.app', 'staging.bsky.app'
            }

            # Filter out common media/CDN domains that aren't content
            media_domains = {
                'pbs.twimg.com', 'imgur.com', 'i.imgur.com',
                'media.tenor.com', 'giphy.com', 'media.giphy.com'
            }

            return any(domain == d or domain.endswith('.' + d) for d in internal_domains | media_domains)

        except:
            return True  # Filter out malformed URLs

    def analyze_top_links(self):
        """Analyze and return top 30 links from relevant clusters"""
        print("  🔗 Analyzing top links from posts...")

        link_stats = defaultdict(lambda: {
            'count': 0,
            'total_likes': 0,
            'posts': [],
            'domain': '',
            'title': ''
        })

        # Process all posts from relevant clusters
        for cluster_id in self.relevant_cluster_ids:
            cluster_posts = self.get_posts_for_cluster(cluster_id)

            for post in cluster_posts:
                links = self.extract_links_from_post(post)

                for link in links:
                    # Normalize URL (remove tracking parameters, etc.)
                    normalized_link = self._normalize_url(link)

                    link_stats[normalized_link]['count'] += 1
                    link_stats[normalized_link]['total_likes'] += post.get('likeCount', 0)
                    link_stats[normalized_link]['posts'].append(post)

                    # Extract domain for grouping
                    try:
                        domain = urlparse(normalized_link).netloc.lower()
                        if domain.startswith('www.'):
                            domain = domain[4:]
                        link_stats[normalized_link]['domain'] = domain
                    except:
                        link_stats[normalized_link]['domain'] = 'unknown'

        # Sort by count and return top 30
        top_links = sorted(
            link_stats.items(),
            key=lambda x: (x[1]['count'], x[1]['total_likes']),
            reverse=True
        )[:30]

        print(f"    📊 Found {len(link_stats)} unique links, showing top 30")
        return top_links

    def _normalize_url(self, url):
        """Normalize URL by removing common tracking parameters"""
        try:
            parsed = urlparse(url)
            # Remove common tracking parameters
            query_parts = []
            if parsed.query:
                for param in parsed.query.split('&'):
                    if '=' in param:
                        key = param.split('=')[0].lower()
                        # Keep non-tracking parameters
                        if key not in {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                                     'fbclid', 'gclid', 'ref', 'referrer', '_hsenc', '_hsmi'}:
                            query_parts.append(param)

            # Reconstruct URL
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if query_parts:
                normalized += '?' + '&'.join(query_parts)

            return normalized

        except:
            return url  # Return original if parsing fails

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
        """Generate screenshots for top 3 posts in a cluster, with fallback to next best posts if screenshots fail"""
        print(f"  📸 Getting screenshots for cluster {cluster_id}...")

        # Get more posts than needed to allow for fallbacks
        top_posts = self.get_top_posts_for_cluster(cluster_id, 10)  # Get up to 10 posts as candidates
        screenshot_info = []
        urls_to_screenshot = []

        # First pass: identify existing screenshots and URLs to generate
        candidate_posts = []
        for post in top_posts:
            url = self.url_from_post(post)
            if not url:
                continue  # Skip posts without valid URLs

            author_handle = post.get('author', {}).get('handle', 'unknown')
            uri = post.get('uri', '')
            post_id = uri.split('/app.bsky.feed.post/')[-1] if '/app.bsky.feed.post/' in uri else 'unknown'

            screenshot_filename = f"bluesky_post_{author_handle}_{post_id}.png"
            screenshot_path = self.screenshots_dir / screenshot_filename

            candidate_posts.append({
                'post': post,
                'url': url,
                'screenshot_path': screenshot_path,
                'screenshot_filename': screenshot_filename,
                'score': self.calculate_post_score(post),
                'exists': screenshot_path.exists()
            })

        # Second pass: intelligently select which screenshots to generate
        # First, see how many of the top 3 already exist
        existing_count = sum(1 for c in candidate_posts[:3] if c['exists'])
        needed_count = 3 - existing_count

        # Only generate screenshots for posts we'll actually use
        urls_to_screenshot = []
        selected_for_generation = []

        if needed_count > 0:
            # Go through candidates in order and select ones that need generation
            for candidate in candidate_posts:
                if not candidate['exists'] and len(selected_for_generation) < needed_count:
                    urls_to_screenshot.append(candidate['url'])
                    selected_for_generation.append(candidate)

        if urls_to_screenshot:
            print(f"    📸 Generating {len(urls_to_screenshot)} screenshots (need {needed_count} total)...")
            print(f"    🔗 URLs: {urls_to_screenshot[:3]}...")  # Show first 3 URLs for debugging
            success = self.generate_screenshots_batch(urls_to_screenshot)

            # Update existence status for ALL candidates after generation attempt
            # (screenshots might have been generated for posts we didn't explicitly request)
            for candidate in candidate_posts:
                candidate['exists'] = candidate['screenshot_path'].exists()

            # Count how many we have now
            total_existing = sum(1 for c in candidate_posts if c['exists'])

            if total_existing < 3:
                # Some screenshots failed, try more candidates
                still_needed = 3 - total_existing
                remaining_candidates = [c for c in candidate_posts if not c['exists']]

                if remaining_candidates:
                    additional_urls = []
                    additional_candidates = []

                    for candidate in remaining_candidates[:still_needed]:
                        additional_urls.append(candidate['url'])
                        additional_candidates.append(candidate)

                    if additional_urls:
                        print(f"    🔄 Some screenshots failed, trying {len(additional_urls)} more...")
                        success = self.generate_screenshots_batch(additional_urls)

                        # Update existence status for ALL candidates again
                        for candidate in candidate_posts:
                            candidate['exists'] = candidate['screenshot_path'].exists()

        # Third pass: select the best 3 posts that have successful screenshots
        successful_screenshots = [c for c in candidate_posts if c['exists']]

        # If we don't have 3 successful screenshots, include failed ones to maintain count
        if len(successful_screenshots) < 3:
            failed_screenshots = [c for c in candidate_posts if not c['exists']]
            # Add failed ones to reach 3 total
            needed = 3 - len(successful_screenshots)
            successful_screenshots.extend(failed_screenshots[:needed])

        # Take the top 3 (prioritizing successful screenshots)
        final_screenshots = successful_screenshots[:3]

        # Convert to the expected format
        for candidate in final_screenshots:
            screenshot_info.append({
                'post': candidate['post'],
                'url': candidate['url'],
                'screenshot_path': candidate['screenshot_path'],
                'screenshot_filename': candidate['screenshot_filename'],
                'score': candidate['score']
            })

        print(f"    ✅ Selected {len(screenshot_info)} posts for cluster {cluster_id}")
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
        html_content += self._generate_table_of_contents()  # Add TOC after title
        html_content += self._generate_methodology_section()  # Methodology back at top
        html_content += self._generate_executive_summary()
        html_content += self._generate_cluster_summary_table()
        html_content += self._generate_individual_cluster_sections()
        html_content += self._generate_author_profiles_section()  # New author profiles section
        html_content += self._generate_special_search_section()  # Special search section
        html_content += self._generate_top_links_section()  # Top shared links section
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
        html {
            scroll-behavior: smooth;
        }
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
<h1 id="top">Bluesky Cluster Analysis Report - {self.session_name.title()}</h1>
"""

    def _generate_table_of_contents(self):
        """Generate clickable table of contents"""
        # Count relevant clusters for individual cluster sections
        num_relevant_clusters = len(self.relevant_cluster_ids)

        # Count search groups
        num_search_groups = len(self.search_config.get('search_groups', []))

        return f"""
<div style="background: #f8f9fa; border: 2px solid #4a9eff; border-radius: 10px; padding: 25px; margin: 30px 0;">
    <h2 style="margin-top: 0; color: #4a9eff;">📋 Table of Contents</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
        <div>
            <h3 style="color: #2c5aa0; margin-bottom: 15px;">📊 Overview & Analysis</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin: 8px 0;"><a href="#methodology" style="color: #4a9eff; text-decoration: none;">📋 Methodology & Data Collection</a></li>
                <li style="margin: 8px 0;"><a href="#executive-summary" style="color: #4a9eff; text-decoration: none;">📊 Executive Summary</a></li>
                <li style="margin: 8px 0;"><a href="#cluster-summary" style="color: #4a9eff; text-decoration: none;">📋 Relevant Clusters Summary</a></li>
            </ul>
        </div>

        <div>
            <h3 style="color: #2c5aa0; margin-bottom: 15px;">🎯 Detailed Sections</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin: 8px 0;"><a href="#individual-clusters" style="color: #4a9eff; text-decoration: none;">🎯 Individual Cluster Analysis ({num_relevant_clusters} clusters)</a></li>
                <li style="margin: 8px 0;"><a href="#author-profiles" style="color: #4a9eff; text-decoration: none;">👤 Top Author Profiles (20 authors)</a></li>
                <li style="margin: 8px 0;"><a href="#special-searches" style="color: #4a9eff; text-decoration: none;">🔍 Special Topic Searches ({num_search_groups} topics)</a></li>
                <li style="margin: 8px 0;"><a href="#top-links" style="color: #4a9eff; text-decoration: none;">🔗 Top Shared Links (30 links)</a></li>
            </ul>
        </div>

        <div>
            <h3 style="color: #2c5aa0; margin-bottom: 15px;">📈 Additional Analysis</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin: 8px 0;"><a href="#irrelevant-clusters" style="color: #4a9eff; text-decoration: none;">🗂️ Irrelevant Clusters</a></li>
                <li style="margin: 8px 0;"><a href="#temporal-analysis" style="color: #4a9eff; text-decoration: none;">📈 Temporal Analysis</a></li>
            </ul>
        </div>
    </div>

    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #ddd; font-size: 0.9em; color: #666;">
        💡 <strong>Navigation Tip:</strong> Click any section title to jump directly to that part of the report.
        Use your browser's "Find" function (Ctrl/Cmd+F) to search for specific terms.
    </div>
</div>
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
<div class="metadata" id="methodology">
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
<div style="text-align: right; margin-bottom: -10px;">
    <small><a href="#top" style="color: #666; text-decoration: none;">↑ Back to top</a></small>
</div>
<h2 id="executive-summary">📊 Executive Summary</h2>

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
<div style="margin: 60px 0 40px 0; border-top: 3px solid #999; padding-top: 40px;" id="temporal-analysis">
<div style="text-align: right; margin-bottom: -10px;">
    <small><a href="#top" style="color: #666; text-decoration: none;">↑ Back to top</a></small>
</div>
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
<div style="text-align: right; margin-bottom: -10px;">
    <small><a href="#top" style="color: #666; text-decoration: none;">↑ Back to top</a></small>
</div>
<h2 id="cluster-summary">📋 Relevant Clusters Summary</h2>
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
            description = cluster_desc.get('title', cluster_desc.get('theme', 'No description available'))
            display_id = self.get_display_cluster_id(cluster_id)

            html += f"""
            <tr>
                <td><strong><a href="#cluster-{display_id}" style="color: #2c5aa0; text-decoration: none;">Cluster {display_id}</a></strong></td>
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

        # Pre-calculate top 20 overall authors for linking
        overall_author_stats = defaultdict(lambda: {
            'posts': 0,
            'likes': 0,
            'handle': '',
            'display_name': ''
        })

        for overall_cluster_id in self.relevant_cluster_ids:
            overall_cluster_posts = self.get_posts_for_cluster(overall_cluster_id)
            for post in overall_cluster_posts:
                author = post.get('author', {})
                author_handle = author.get('handle', 'unknown')
                overall_author_stats[author_handle]['posts'] += 1
                overall_author_stats[author_handle]['likes'] += post.get('likeCount', 0)
                overall_author_stats[author_handle]['handle'] = author_handle
                overall_author_stats[author_handle]['display_name'] = author.get('displayName', author_handle)

        # Find qualified authors (same logic as author profiles)
        cluster_top_authors = set()
        for qual_cluster_id in self.relevant_cluster_ids:
            qual_cluster_posts = self.get_posts_for_cluster(qual_cluster_id)
            qual_cluster_author_stats = defaultdict(lambda: {'posts': 0, 'likes': 0, 'handle': ''})
            for post in qual_cluster_posts:
                author = post.get('author', {})
                author_handle = author.get('handle', 'unknown')
                qual_cluster_author_stats[author_handle]['posts'] += 1
                qual_cluster_author_stats[author_handle]['likes'] += post.get('likeCount', 0)

            top_2_cluster_authors = sorted(
                qual_cluster_author_stats.items(),
                key=lambda x: x[1]['likes'],
                reverse=True
            )[:2]

            for author_handle, _ in top_2_cluster_authors:
                cluster_top_authors.add(author_handle)

        qualified_overall_authors = [
            (handle, stats) for handle, stats in overall_author_stats.items()
            if handle in cluster_top_authors
        ]

        top_20_overall = sorted(qualified_overall_authors, key=lambda x: x[1]['likes'], reverse=True)[:20]
        self.top_20_handles_with_ranks = {handle: rank for rank, (handle, _) in enumerate(top_20_overall, 1)}

        html = """
<div style="margin: 60px 0 40px 0; border-top: 3px solid #4a9eff; padding-top: 40px;" id="individual-clusters">
<div style="text-align: right; margin-bottom: -10px;">
    <small><a href="#top" style="color: #666; text-decoration: none;">↑ Back to top</a></small>
</div>
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
<div class="cluster-card" id="cluster-{display_id}">
    <h2 style="color: #4a9eff; font-size: 1.8em; margin-bottom: 20px;">Cluster {display_id}</h2>

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

    <h4>📝 {cluster_desc.get('title', 'Cluster Theme')}</h4>
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

                # Check if this author is in the overall top 20
                author_cell = f"<strong>{display_name}</strong><br>"
                if handle in self.top_20_handles_with_ranks:
                    profile_rank = self.top_20_handles_with_ranks[handle]
                    author_cell += f"<small><a href=\"#author-profile-{profile_rank}\" style=\"color: #8e44ad; text-decoration: none;\">📊 Top {profile_rank} Overall</a> | <a href=\"https://bsky.app/profile/{handle}\" target=\"_blank\">@{handle}</a></small>"
                else:
                    author_cell += f"<small><a href=\"https://bsky.app/profile/{handle}\" target=\"_blank\">@{handle}</a></small>"

                html += f"""
            <tr>
                <td>{rank}</td>
                <td>{author_cell}</td>
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

    def _generate_author_profiles_section(self):
        """Generate individual author profile sections for top 20 authors"""
        print("  👤 Generating author profiles section...")

        # Calculate overall stats for relevant clusters only (same logic as executive summary)
        author_stats = defaultdict(lambda: {
            'posts': 0,
            'likes': 0,
            'handle': '',
            'display_name': '',
            'posts_data': [],
            'cluster_activity': defaultdict(int)  # cluster_id -> post_count
        })

        for cluster_id in self.relevant_cluster_ids:
            cluster_posts = self.get_posts_for_cluster(cluster_id)

            for post in cluster_posts:
                # Author stats
                author = post.get('author', {})
                author_handle = author.get('handle', 'unknown')
                author_stats[author_handle]['posts'] += 1
                author_stats[author_handle]['likes'] += post.get('likeCount', 0)
                author_stats[author_handle]['handle'] = author_handle
                author_stats[author_handle]['display_name'] = author.get('displayName', author_handle)
                author_stats[author_handle]['posts_data'].append(post)

                # Track cluster activity
                author_stats[author_handle]['cluster_activity'][cluster_id] += 1

        # Find authors who appear in top 5 of any cluster
        cluster_top_authors = set()
        for cluster_id in self.relevant_cluster_ids:
            cluster_posts = self.get_posts_for_cluster(cluster_id)

            # Get top authors for this cluster
            cluster_author_stats = defaultdict(lambda: {'posts': 0, 'likes': 0, 'handle': ''})
            for post in cluster_posts:
                author = post.get('author', {})
                author_handle = author.get('handle', 'unknown')
                cluster_author_stats[author_handle]['posts'] += 1
                cluster_author_stats[author_handle]['likes'] += post.get('likeCount', 0)
                cluster_author_stats[author_handle]['handle'] = author_handle

            # Get top 2 authors in this cluster by total likes
            top_2_cluster_authors = sorted(
                cluster_author_stats.items(),
                key=lambda x: x[1]['likes'],
                reverse=True
            )[:2]

            # Add these authors to our qualified set
            for author_handle, _ in top_2_cluster_authors:
                cluster_top_authors.add(author_handle)

        print(f"    📊 Found {len(cluster_top_authors)} authors who appear in top 2 of at least one cluster")

        # Filter overall top authors to only include those who appear in cluster top 2s
        qualified_authors = [
            (handle, stats) for handle, stats in author_stats.items()
            if handle in cluster_top_authors
        ]

        # Top 20 qualified authors by total likes
        top_authors = sorted(qualified_authors, key=lambda x: x[1]['likes'], reverse=True)[:20]

        print(f"    🏆 Selected {len(top_authors)} top authors (qualified by cluster top-2 appearances)")

        html = f"""
<div style="margin: 60px 0 40px 0; border-top: 3px solid #8e44ad; padding-top: 40px;" id="author-profiles">
<div style="text-align: right; margin-bottom: -10px;">
    <small><a href="#top" style="color: #666; text-decoration: none;">↑ Back to top</a></small>
</div>
<h2>👤 Top Author Profiles</h2>
<p>Individual profiles for the top 20 authors by total likes, featuring screenshots of their most engaging posts.</p>

<h3>🏆 Top 20 Authors Overview (by total likes in relevant clusters)</h3>
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
            <td><strong><a href="#author-profile-{i}" style="color: #8e44ad; text-decoration: none;">{display_name}</a></strong><br><small><a href="https://bsky.app/profile/{handle}" target="_blank">@{handle}</a></small></td>
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

        # Generate profile for each top author
        for rank, (handle, stats) in enumerate(top_authors, 1):
            display_name = stats['display_name'] if stats['display_name'] != handle else handle
            avg_likes_per_post = stats['likes'] / stats['posts'] if stats['posts'] > 0 else 0

            print(f"    👤 Processing profile for {display_name} (@{handle}) - Rank {rank}")

            # Get top posts by this author (up to 12)
            author_posts = stats['posts_data']

            # Sort by like count and take top 9
            top_posts = sorted(author_posts, key=lambda p: p.get('likeCount', 0), reverse=True)[:9]

            # Generate screenshots for top posts
            screenshot_info = self._generate_screenshots_for_author(handle, top_posts)

            # Get all clusters for this author for counting, but limit display to top 5
            cluster_activity = stats['cluster_activity']
            all_clusters = sorted(cluster_activity.items(), key=lambda x: x[1], reverse=True)
            top_5_clusters = all_clusters[:5]

            # Create cluster activity HTML
            cluster_activity_html = ""
            if top_5_clusters:
                cluster_activity_html = "<h4>🎯 Most Active Clusters</h4><div class=\"keywords\">"
                for cluster_id, post_count in top_5_clusters:
                    display_id = self.get_display_cluster_id(cluster_id)
                    cluster_desc = self.cluster_descriptions[cluster_id]
                    cluster_title = cluster_desc.get('title', cluster_desc.get('theme', 'Unknown theme')[:50])  # Use title, fallback to truncated theme
                    cluster_activity_html += f'<span class="keyword" style="background: #e8f4fd; border-color: #8e44ad;"><a href="#cluster-{display_id}" style="color: #8e44ad; text-decoration: none;">Cluster {display_id}: {post_count} posts</a><br><small>{cluster_title}</small></span>'
                cluster_activity_html += "</div>"

            html += f"""
<div class="cluster-card" style="border-left: 4px solid #8e44ad;" id="author-profile-{rank}">
    <h2 style="color: #8e44ad; font-size: 1.8em; margin-bottom: 20px;">#{rank}: {display_name}</h2>
    <p><a href="https://bsky.app/profile/{handle}" target="_blank" style="color: #8e44ad;">@{handle}</a></p>

    <div class="summary-stats">
        <div class="stat-card">
            <div class="stat-number">{stats['posts']:,}</div>
            <div class="stat-label">Posts</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{stats['likes']:,}</div>
            <div class="stat-label">Total Likes</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{avg_likes_per_post:.1f}</div>
            <div class="stat-label">Avg Likes/Post</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(all_clusters)}</div>
            <div class="stat-label">Active Clusters</div>
        </div>
    </div>

    {cluster_activity_html}

    <h4>📸 Top Posts (up to 9)</h4>
    <div class="screenshots" style="grid-template-columns: repeat(3, 1fr); gap: 15px;">
"""

            # Add screenshots
            for i, screenshot in enumerate(screenshot_info, 1):
                post = screenshot['post']
                likes = post.get('likeCount', 0)

                # Get cluster information for this post
                uri = post.get('uri')
                cluster_id = None
                display_cluster_id = None
                cluster_title = 'Unknown cluster'

                if uri and uri in self.cluster_assignments:
                    cluster_id = self.cluster_assignments[uri].get('cluster_id')
                    if cluster_id is not None:
                        display_cluster_id = self.get_display_cluster_id(cluster_id)
                        cluster_desc = self.cluster_descriptions.get(cluster_id, {})
                        cluster_title = cluster_desc.get('title', cluster_desc.get('theme', 'Unknown theme'))

                # Convert screenshot to base64 for embedding
                base64_image = self.screenshot_to_base64(screenshot['screenshot_path'])

                if base64_image:
                    html += f"""
        <div class="screenshot">
            <img src="{base64_image}" alt="Post {i} by {display_name}">
            <div style="padding: 10px; background: #f8f9fa;">
                <strong>Post #{i}</strong><br>
                <small>{likes:,} likes</small><br>"""

                    # Add cluster info if available
                    if cluster_id is not None and display_cluster_id is not None:
                        html += f"""
                <div style="margin: 5px 0;">
                    <small style="background: #e3f2fd; padding: 2px 6px; border-radius: 3px; color: #1976d2;">
                        <a href="#cluster-{display_cluster_id}" style="text-decoration: none; color: #1976d2;">
                            Cluster {display_cluster_id}: {cluster_title}
                        </a>
                    </small>
                </div>"""

                    html += f"""
                <small><a href="{screenshot['url']}" target="_blank">View original</a></small>
            </div>
        </div>
"""
                else:
                    html += f"""
        <div class="screenshot">
            <div style="padding: 20px; background: #f8f9fa; text-align: center; color: #666;">
                <p><strong>Screenshot not available</strong></p>
                <p>Post #{i}: {likes:,} likes</p>"""

                    # Add cluster info if available
                    if cluster_id is not None and display_cluster_id is not None:
                        html += f"""
                <div style="margin: 5px 0;">
                    <small style="background: #e3f2fd; padding: 2px 6px; border-radius: 3px; color: #1976d2;">
                        <a href="#cluster-{display_cluster_id}" style="text-decoration: none; color: #1976d2;">
                            Cluster {display_cluster_id}: {cluster_title}
                        </a>
                    </small>
                </div>"""

                    html += f"""
                <small><a href="{screenshot['url']}" target="_blank">View original</a></small>
            </div>
        </div>
"""

            html += """
    </div>
</div>
"""

        return html

    def _generate_screenshots_for_author(self, author_handle: str, posts: list):
        """Generate screenshots for an author's top posts"""
        print(f"    📸 Getting screenshots for @{author_handle}...")

        screenshot_info = []
        urls_to_screenshot = []

        for i, post in enumerate(posts, 1):
            url = self.url_from_post(post)
            if not url:
                print(f"    ⚠️  Could not generate URL for post {i}")
                continue

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
                })
            else:
                urls_to_screenshot.append(url)
                screenshot_info.append({
                    'post': post,
                    'url': url,
                    'screenshot_path': screenshot_path,
                    'screenshot_filename': screenshot_filename,
                })

        # Generate missing screenshots if any
        if urls_to_screenshot:
            print(f"    📸 Generating {len(urls_to_screenshot)} screenshots for @{author_handle}...")
            self.generate_screenshots_batch(urls_to_screenshot)

        return screenshot_info

    def _generate_top_links_section(self):
        """Generate top links section showing the most shared links"""
        print("  🔗 Generating top links section...")

        top_links = self.analyze_top_links()

        if not top_links:
            return """
<div style="margin: 60px 0 40px 0; border-top: 3px solid #ff6b6b; padding-top: 40px;">
<h2>🔗 Top Shared Links</h2>
<p>No external links found in the analyzed posts.</p>
</div>
"""

        # Group links by domain for summary
        domain_stats = defaultdict(int)
        for url, stats in top_links:
            domain_stats[stats['domain']] += stats['count']

        top_domains = sorted(domain_stats.items(), key=lambda x: x[1], reverse=True)[:10]

        html = f"""
<div style="margin: 60px 0 40px 0; border-top: 3px solid #ff6b6b; padding-top: 40px;" id="top-links">
<div style="text-align: right; margin-bottom: -10px;">
    <small><a href="#top" style="color: #666; text-decoration: none;">↑ Back to top</a></small>
</div>
<h2>🔗 Top Shared Links</h2>
<p>The most frequently shared external links across all relevant clusters, showing the content and resources that the community is engaging with.</p>

<h3>📊 Top Domains</h3>
<div class="summary-stats">
"""

        # Add top domains stats
        for i, (domain, count) in enumerate(top_domains[:5]):
            html += f"""
    <div class="stat-card">
        <div class="stat-number">{count}</div>
        <div class="stat-label"><a href="https://{domain}" target="_blank" style="color: #4a9eff; text-decoration: none;">{domain}</a></div>
    </div>
"""

        html += """
</div>

<h3>🔗 Top 30 Shared Links</h3>
<table>
    <thead>
        <tr>
            <th>Rank</th>
            <th>Link</th>
            <th>Domain</th>
            <th>Shares</th>
            <th>Total Likes</th>
            <th>Avg Likes/Share</th>
        </tr>
    </thead>
    <tbody>
"""

        # Add individual links
        for i, (url, stats) in enumerate(top_links, 1):
            # Truncate long URLs for display
            display_url = url
            if len(display_url) > 60:
                display_url = display_url[:57] + "..."

            avg_likes = stats['total_likes'] / stats['count'] if stats['count'] > 0 else 0

            html += f"""
        <tr>
            <td>{i}</td>
            <td><a href="{url}" target="_blank" style="color: #4a9eff;">{display_url}</a></td>
            <td>{stats['domain']}</td>
            <td>{stats['count']}</td>
            <td>{stats['total_likes']:,}</td>
            <td>{avg_likes:.1f}</td>
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
<div style="margin: 60px 0 40px 0; border-top: 3px solid #999; padding-top: 40px;" id="irrelevant-clusters">
<div style="text-align: right; margin-bottom: -10px;">
    <small><a href="#top" style="color: #666; text-decoration: none;">↑ Back to top</a></small>
</div>
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
            title = cluster_desc.get('title', 'Untitled Cluster')
            theme = cluster_desc.get('theme', 'No description available')

            html += f"""
            <tr>
                <td>Cluster {i}</td>
                <td>{post_count:,}</td>
                <td><strong>{title}</strong><br><span style="color: #666; font-size: 0.9em;">{theme}</span></td>
            </tr>
"""

        html += """
        </tbody>
    </table>
</div>
"""

        return html


    def _generate_special_search_section(self):
        """Generate special search section using pre-generated special topics data"""
        print("  🔍 Generating special search section...")

        # Load special topics data
        special_topics_file = self.datasets_dir / f"{self.session_name}_special_topics.jsonl"

        if not special_topics_file.exists():
            return """
<div style="margin: 60px 0 40px 0; border-top: 3px solid #9b59b6; padding-top: 40px;" id="special-searches">
<h2>🔍 Special Topic Searches</h2>
<p>Special topics analysis not found. Run <code>python special_topics.py {session_name}</code> to generate this section.</p>
</div>
""".format(session_name=self.session_name)

        special_topics = []
        try:
            with open(special_topics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        special_topics.append(json.loads(line.strip()))
        except Exception as e:
            print(f"    ❌ Error loading special topics data: {e}")
            return """
<div style="margin: 60px 0 40px 0; border-top: 3px solid #9b59b6; padding-top: 40px;" id="special-searches">
<h2>🔍 Special Topic Searches</h2>
<p>Error loading special topics data. Please regenerate with <code>python special_topics.py {session_name}</code>.</p>
</div>
""".format(session_name=self.session_name)

        if not special_topics:
            return """
<div style="margin: 60px 0 40px 0; border-top: 3px solid #9b59b6; padding-top: 40px;" id="special-searches">
<h2>🔍 Special Topic Searches</h2>
<p>No special topics found in data file.</p>
</div>
"""

        html = f"""
<div style="margin: 60px 0 40px 0; border-top: 3px solid #9b59b6; padding-top: 40px;" id="special-searches">
<div style="text-align: right; margin-bottom: -10px;">
    <small><a href="#top" style="color: #666; text-decoration: none;">↑ Back to top</a></small>
</div>
<h2>🔍 Special Topic Searches</h2>
<p>Deep dives into specific topics and themes, showcasing the top 6 posts for each search category with AI-generated summaries.</p>
</div>
"""

        # Process each special topic
        for group_idx, topic in enumerate(special_topics, 1):
            title = topic.get('topic_title', f'Topic {group_idx}')
            keywords = topic.get('keywords', [])
            posts = topic.get('posts', [])
            ai_summary = topic.get('ai_summary', 'No summary available.')

            print(f"    📝 Processing topic: {title} ({len(posts)} posts)")

            # Generate screenshots for posts
            screenshot_info = self._generate_screenshots_for_special_topic_posts(title, posts)

            # Format keywords using the same style as methodology section
            keywords_html = '<div class="keywords">'
            for keyword in keywords:
                keywords_html += f'<span class="keyword">{keyword}</span>'
            keywords_html += '</div>'

            # Get total post count for display
            total_count = topic.get('total_post_count', len(posts))
            display_text = f"{total_count} posts found"
            if total_count > len(posts):
                display_text += f" (showing top {len(posts)} by engagement score)"
            elif len(posts) > 0:
                display_text += f" (showing top {len(posts)} by engagement score)"

            html += f"""
<div class="cluster-card" style="border-left: 4px solid #9b59b6;" id="search-{group_idx}">
    <h2 style="color: #9b59b6; font-size: 1.8em; margin-bottom: 20px;">{title}</h2>
    <p><strong>Search Keywords:</strong></p>
    {keywords_html}

    <h4>🤖 AI Analysis</h4>
    <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #9b59b6; margin: 15px 0;">
        <p style="margin: 0; font-style: italic; color: #333;">{ai_summary}</p>
    </div>

    <p><strong>Results:</strong> {display_text}</p>

    <h4>📸 Top Posts</h4>
    <div class="screenshots" style="grid-template-columns: repeat(3, 1fr); gap: 20px;">
"""

            # Add screenshots
            for i, screenshot in enumerate(screenshot_info, 1):
                post_data = screenshot['post_data']
                author = post_data.get('author', {})
                author_name = author.get('displayName', author.get('handle', 'Unknown'))
                author_handle = author.get('handle', 'unknown')
                likes = post_data.get('likeCount', 0)
                cluster_id = post_data.get('cluster_id')
                display_cluster_id = post_data.get('display_cluster_id')

                # Get cluster info
                cluster_desc = self.cluster_descriptions.get(cluster_id, {})
                cluster_title = cluster_desc.get('title', cluster_desc.get('theme', 'Unknown theme'))

                # Convert screenshot to base64 for embedding
                base64_image = self.screenshot_to_base64(screenshot['screenshot_path'])

                # Get semantic content for alt text
                semantic_content = post_data.get('semantic_content', '')
                # Clean semantic content for alt text (no truncation)
                alt_text = semantic_content.replace('"', '&quot;').replace('\n', ' ').strip()

                if base64_image:
                    html += f"""
        <div class="screenshot">
            <img src="{base64_image}" alt="{alt_text}">
            <div style="padding: 15px; background: #f8f9fa;">
                <div style="margin-bottom: 8px;">
                    <strong>#{i}: {author_name}</strong>
                </div>
                <div style="margin-bottom: 8px;">
                    <small style="color: #666;">@{author_handle}</small>
                </div>
                <div style="margin-bottom: 8px;">
                    <small style="background: #e3f2fd; padding: 2px 6px; border-radius: 3px; color: #1976d2;">
                        <a href="#cluster-{display_cluster_id}" style="text-decoration: none; color: #1976d2;">
                            Cluster {display_cluster_id}: {cluster_title}
                        </a>
                    </small>
                </div>
                <div>
                    <small><a href="{screenshot['url']}" target="_blank" style="color: #9b59b6;">View original →</a></small>
                </div>
            </div>
        </div>
"""
                else:
                    html += f"""
        <div class="screenshot">
            <div style="padding: 20px; background: #f8f9fa; text-align: center; color: #666; min-height: 200px; display: flex; align-items: center; justify-content: center;">
                <div>
                    <p><strong>Screenshot not available</strong></p>
                    <p>#{i}: {author_name}</p>"""

                    # Add cluster info if available
                    if cluster_id is not None and display_cluster_id is not None:
                        html += f"""
                    <div style="margin: 5px 0;">
                        <small style="background: #e3f2fd; padding: 2px 6px; border-radius: 3px; color: #1976d2;">
                            <a href="#cluster-{display_cluster_id}" style="text-decoration: none; color: #1976d2;">
                                Cluster {display_cluster_id}: {cluster_title}
                            </a>
                        </small>
                    </div>"""

                    html += f"""
                    <small><a href="{screenshot['url']}" target="_blank" style="color: #9b59b6;">View original →</a></small>
                </div>
            </div>
        </div>
"""

            html += """
    </div>
</div>
"""

        return html

    def _generate_screenshots_for_special_topic_posts(self, topic_title: str, posts: list):
        """Generate screenshots for special topic posts"""
        print(f"    📸 Getting screenshots for topic: {topic_title}...")

        screenshot_info = []
        urls_to_screenshot = []

        for i, post_data in enumerate(posts, 1):
            uri = post_data.get('uri')
            if not uri:
                continue

            # Convert to Bluesky URL
            author_handle = post_data.get('author', {}).get('handle', 'unknown')
            if '/app.bsky.feed.post/' in uri:
                post_id = uri.split('/app.bsky.feed.post/')[-1]
                url = f"https://bsky.app/profile/{author_handle}/post/{post_id}"
            else:
                continue

            screenshot_filename = f"bluesky_post_{author_handle}_{post_id}.png"
            screenshot_path = self.screenshots_dir / screenshot_filename

            # Check if screenshot already exists
            if screenshot_path.exists():
                screenshot_info.append({
                    'post_data': post_data,
                    'url': url,
                    'screenshot_path': screenshot_path,
                    'screenshot_filename': screenshot_filename
                })
            else:
                urls_to_screenshot.append(url)
                screenshot_info.append({
                    'post_data': post_data,
                    'url': url,
                    'screenshot_path': screenshot_path,
                    'screenshot_filename': screenshot_filename
                })

        # Generate missing screenshots if any
        if urls_to_screenshot:
            print(f"    📸 Generating {len(urls_to_screenshot)} screenshots for topic: {topic_title}...")
            self.generate_screenshots_batch(urls_to_screenshot)

        return screenshot_info


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