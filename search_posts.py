#!/usr/bin/env python3
"""
Bluesky Posts Keyword Search Tool
Usage: python search_posts.py <session_name> <keyword> [--limit N] [--min-likes N] [--sort-by likes|score]
Example: python search_posts.py july "artificial intelligence" --limit 20 --min-likes 10
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime


class PostSearcher:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.datasets_dir = Path('datasets')

        # File paths
        self.cluster_descriptions_file = self.datasets_dir / f"{session_name}_cluster_descriptions.jsonl"
        self.clusters_file = self.datasets_dir / f"{session_name}_clusters.jsonl"
        self.processed_file = self.datasets_dir / f"{session_name}_processed.jsonl"

        # Data containers
        self.cluster_descriptions = {}
        self.cluster_assignments = {}
        self.posts = {}
        self.relevant_cluster_ids = set()
        self.cluster_id_mapping = {}

    def load_data(self):
        """Load all required data files"""
        print(f"📊 Loading data for session: {self.session_name}")

        # Load cluster descriptions
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

        print(f"  ✅ Loaded {len(self.cluster_descriptions)} cluster descriptions")
        print(f"  ✅ Found {len(self.relevant_cluster_ids)} relevant clusters")

        # Load cluster assignments
        if not self.clusters_file.exists():
            raise FileNotFoundError(f"Cluster assignments file not found: {self.clusters_file}")

        with open(self.clusters_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                uri = data['uri']
                self.cluster_assignments[uri] = data

        print(f"  ✅ Loaded {len(self.cluster_assignments)} cluster assignments")

        # Load processed posts
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

        print(f"  ✅ Loaded {len(self.posts):,} processed posts")

        # Create cluster ID mapping
        self._create_cluster_id_mapping()

        return True

    def _create_cluster_id_mapping(self):
        """Create mapping from original cluster IDs to sequential numbers for relevant clusters only"""
        relevant_clusters_by_size = []
        for cluster_id, cluster_desc in self.cluster_descriptions.items():
            is_relevant = cluster_desc.get('relevance') == 'relevant'
            if is_relevant:
                post_count = cluster_desc.get('post_count', 0)
                relevant_clusters_by_size.append((post_count, cluster_id))

        # Sort by post count (descending)
        relevant_clusters_by_size.sort(key=lambda x: x[0], reverse=True)

        # Create mapping for relevant clusters only
        for display_number, (post_count, original_id) in enumerate(relevant_clusters_by_size, 1):
            self.cluster_id_mapping[original_id] = display_number

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

    def search_posts(self, keywords: list, limit: int = 10, min_likes: int = 0, sort_by: str = 'likes'):
        """Search for posts containing the keywords and return top results"""
        keywords_str = ', '.join(f"'{kw}'" for kw in keywords)
        print(f"🔍 Searching for keywords: {keywords_str}")
        print(f"  📊 Parameters: limit={limit}, min_likes={min_likes}, sort_by={sort_by}")

        matching_posts = []

        # Search through all posts in relevant clusters
        for cluster_id in self.relevant_cluster_ids:
            cluster_posts = self.get_posts_for_cluster(cluster_id)

            for post in cluster_posts:
                # Skip posts below minimum likes threshold
                if post.get('likeCount', 0) < min_likes:
                    continue

                # Check if any keyword appears in post content
                if self._post_matches_keywords(post, keywords):
                    # Calculate post score: likeCount * probability^2
                    score = post.get('likeCount', 0) * (post.get('cluster_probability', 0) ** 2)

                    matching_posts.append({
                        'post': post,
                        'score': score,
                        'cluster_id': cluster_id,
                        'display_cluster_id': self.get_display_cluster_id(cluster_id)
                    })

        print(f"  ✅ Found {len(matching_posts)} matching posts")

        # Sort results
        if sort_by == 'likes':
            matching_posts.sort(key=lambda x: x['post'].get('likeCount', 0), reverse=True)
        elif sort_by == 'score':
            matching_posts.sort(key=lambda x: x['score'], reverse=True)
        else:
            raise ValueError(f"Invalid sort_by option: {sort_by}. Use 'likes' or 'score'")

        return matching_posts[:limit]

    def _post_matches_keywords(self, post: dict, keywords: list) -> bool:
        """Check if post content matches any of the keywords (case-insensitive)"""
        keywords_lower = [kw.lower() for kw in keywords]

        # Get all text content from post
        text_sources = [
            post.get('record', {}).get('text', ''),
            post.get('semantic_content', ''),
            post.get('parentText', ''),
            post.get('rootText', '')
        ]

        # Combine all text content
        all_text = ' '.join(filter(None, text_sources)).lower()

        # Check if any keyword appears in the combined text
        for keyword in keywords_lower:
            if keyword in all_text:
                return True

        return False

    def url_from_post(self, post):
        """Convert post data to Bluesky URL"""
        author_handle = post.get('author', {}).get('handle', '')
        uri = post.get('uri', '')

        # Extract post ID from AT URI
        if '/app.bsky.feed.post/' in uri:
            post_id = uri.split('/app.bsky.feed.post/')[-1]
            return f"https://bsky.app/profile/{author_handle}/post/{post_id}"

        return None

    def display_results(self, results: list, keywords: list):
        """Display search results in a nicely formatted way"""
        if not results:
            keywords_str = ', '.join(f"'{kw}'" for kw in keywords)
            print(f"❌ No posts found matching {keywords_str}")
            return

        keywords_str = ', '.join(f"'{kw}'" for kw in keywords)
        print(f"\n🎯 Top {len(results)} posts mentioning {keywords_str}:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            post = result['post']
            cluster_id = result['cluster_id']
            display_cluster_id = result['display_cluster_id']
            score = result['score']

            # Post details
            author = post.get('author', {})
            author_name = author.get('displayName', author.get('handle', 'Unknown'))
            author_handle = author.get('handle', 'unknown')
            likes = post.get('likeCount', 0)
            reposts = post.get('repostCount', 0)
            replies = post.get('replyCount', 0)

            # Post content (full semantic content including thread context)
            text = post.get('semantic_content', post.get('record', {}).get('text', ''))

            # Cluster info
            cluster_desc = self.cluster_descriptions.get(cluster_id, {})
            cluster_title = cluster_desc.get('title', cluster_desc.get('theme', 'Unknown theme'))

            # Post URL
            url = self.url_from_post(post)

            print(f"\n#{i}. {author_name} (@{author_handle})")
            print(f"    📊 {likes:,} likes • {reposts:,} reposts • {replies:,} replies")
            print(f"    🏷️  Cluster {display_cluster_id}: {cluster_title}")
            print(f"    💬 \"{text}\"")
            if url:
                print(f"    🔗 {url}")
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='Search posts by keyword')
    parser.add_argument('session_name', help='Name of the session (e.g., july)')
    parser.add_argument('keywords', nargs='+', help='Keywords to search for (one or more)')
    parser.add_argument('--limit', type=int, default=10, help='Number of results to show (default: 10)')
    parser.add_argument('--min-likes', type=int, default=0, help='Minimum likes threshold (default: 0)')
    parser.add_argument('--sort-by', choices=['likes', 'score'], default='likes',
                       help='Sort results by likes or score (default: likes)')

    args = parser.parse_args()

    try:
        searcher = PostSearcher(args.session_name)
        searcher.load_data()

        results = searcher.search_posts(
            keywords=args.keywords,
            limit=args.limit,
            min_likes=args.min_likes,
            sort_by=args.sort_by
        )

        searcher.display_results(results, args.keywords)

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())