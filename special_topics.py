#!/usr/bin/env python3
"""
Special Topics Analysis for Bluesky posts.
Generates AI-powered summaries for configurable search topics.
Usage: python special_topics.py <session_name>
Example: python special_topics.py august
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import os
from dotenv import load_dotenv
from openai import OpenAI


class SpecialTopicsAnalyzer:
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
        self.cluster_id_mapping = {}  # Maps original ID -> new sequential number

        # Search config
        self.search_config = self.load_search_config()

    def load_search_config(self):
        """Load search configuration from search_config.json"""
        config_file = Path('search_config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error: Could not load search config: {e}")
                sys.exit(1)
        else:
            print("❌ Error: search_config.json not found")
            sys.exit(1)

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

    def search_posts_by_keywords(self, keywords: list, limit: int = 6):
        """Search for posts containing the keywords and return top results"""
        matching_posts = []

        # Search through all posts in relevant clusters
        for cluster_id in self.relevant_cluster_ids:
            cluster_posts = self.get_posts_for_cluster(cluster_id)

            for post in cluster_posts:
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

        # Sort by score (likes * probability^2) and return top results
        matching_posts.sort(key=lambda x: x['score'], reverse=True)
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

    def generate_topic_summary(self, topic_title: str, keywords: list, posts: list):
        """Generate AI summary for a topic using GPT-5 mini"""
        print(f"  🤖 Generating AI summary for: {topic_title}")

        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            print("❌ Error: OPENAI_API_KEY must be set in .env file")
            return "Summary generation failed: Missing OpenAI API key"

        client = OpenAI(api_key=api_key)

        # Load prompt template
        prompt_file = Path('prompts/special_topics_summary.txt')
        if not prompt_file.exists():
            print(f"❌ Prompt file not found: {prompt_file}")
            return "Summary generation failed: Missing prompt template"

        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        # Format posts content for prompt
        posts_content = []
        for i, post in enumerate(posts, 1):
            semantic_content = post.get('semantic_content', post.get('record', {}).get('text', ''))
            posts_content.append(f"{i}. {semantic_content}")

        posts_text = "\n".join(posts_content)
        print(f"    📊 Total content length being sent to AI: {len(posts_text)} characters")

        # Format prompt (only posts_content, no topic or keywords to avoid anchoring)
        prompt = prompt_template.replace("{posts_content}", posts_text)

        try:
            # Call GPT-5 mini
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000
            )

            response_text = response.choices[0].message.content
            if response_text is None:
                response_text = ""
            response_text = response_text.strip()

            # Parse JSON response
            try:
                ai_analysis = json.loads(response_text)
                return ai_analysis.get('summary', 'Summary parsing failed')
            except json.JSONDecodeError as e:
                print(f"🚨 CRITICAL ERROR: Invalid JSON response for topic {topic_title}")
                print(f"🚨 JSON Error: {e}")
                print(f"🚨 Raw response: {response_text[:200]}...")
                return f"Summary generation failed: Invalid JSON response - {str(e)}"

        except Exception as e:
            print(f"🚨 CRITICAL ERROR: Failed to generate summary for topic {topic_title}")
            print(f"🚨 Error details: {e}")
            return f"Summary generation failed: {str(e)}"

    def analyze_special_topics(self):
        """Analyze all special topics and generate summaries"""
        print("🔍 Analyzing special topics...")

        search_groups = self.search_config.get('search_groups', [])
        if not search_groups:
            print("❌ No search groups found in search_config.json")
            return []

        results = []

        for group_idx, search_group in enumerate(search_groups, 1):
            topic_title = search_group.get('title', f'Topic {group_idx}')
            keywords = search_group.get('keywords', [])

            if not keywords:
                print(f"⚠️  Skipping {topic_title}: No keywords provided")
                continue

            print(f"  🔍 Processing topic: {topic_title} ({len(keywords)} keywords)")

            # Search for posts (get more for accurate total count, but only use top 6)
            all_search_results = self.search_posts_by_keywords(keywords, limit=1000)  # Get all matching posts
            top_6_results = all_search_results[:6]  # Take top 6 for display and analysis

            if not all_search_results:
                print(f"    ❌ No posts found for {topic_title}")
                results.append({
                    'topic_title': topic_title,
                    'keywords': keywords,
                    'posts': [],
                    'ai_summary': "No posts found matching the search keywords.",
                    'total_post_count': 0
                })
                continue

            total_count = len(all_search_results)
            display_count = len(top_6_results)
            print(f"    ✅ Found {total_count} total posts for {topic_title} (using top {display_count} for analysis)")

            # Extract post data for output (only top 6)
            posts_data = []
            for result in top_6_results:
                post = result['post']
                posts_data.append({
                    'uri': post.get('uri'),
                    'author': post.get('author', {}),
                    'likeCount': post.get('likeCount', 0),
                    'repostCount': post.get('repostCount', 0),
                    'replyCount': post.get('replyCount', 0),
                    'semantic_content': post.get('semantic_content', ''),
                    'cluster_id': result['cluster_id'],
                    'display_cluster_id': result['display_cluster_id'],
                    'score': result['score']
                })

            # Extract posts for AI summary (use top 6 for analysis)
            posts_for_summary = [result['post'] for result in top_6_results]

            # Generate AI summary
            ai_summary = self.generate_topic_summary(topic_title, keywords, posts_for_summary)

            # Print summary to terminal
            print(f"    🤖 AI Summary for '{topic_title}':")
            print(f"    └── {ai_summary}")
            print()

            results.append({
                'topic_title': topic_title,
                'keywords': keywords,
                'posts': posts_data,
                'ai_summary': ai_summary,
                'total_post_count': total_count  # Store total count for sorting
            })

        # Sort results by total number of posts found (descending)
        print("📊 Sorting results by total post count...")
        results.sort(key=lambda x: x.get('total_post_count', 0), reverse=True)

        for i, result in enumerate(results, 1):
            total_count = result.get('total_post_count', 0)
            display_count = len(result['posts'])
            if total_count > display_count:
                print(f"  {i}. {result['topic_title']}: {total_count} posts (showing top {display_count})")
            else:
                print(f"  {i}. {result['topic_title']}: {total_count} posts")

        return results

    def save_results(self, results: list):
        """Save special topics analysis results to JSONL file"""
        output_file = self.datasets_dir / f"{self.session_name}_special_topics.jsonl"

        print(f"💾 Saving results to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(results)} special topics analyses")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Generate special topics analysis with AI summaries')
    parser.add_argument('session_name', help='Name of the session (e.g., "august")')

    args = parser.parse_args()

    print(f"🔍 Analyzing special topics for session: {args.session_name}")
    print("=" * 60)

    try:
        # Initialize analyzer
        analyzer = SpecialTopicsAnalyzer(args.session_name)

        # Load data
        analyzer.load_data()

        # Analyze special topics
        results = analyzer.analyze_special_topics()

        if not results:
            print("❌ No results generated")
            return 1

        # Save results
        output_file = analyzer.save_results(results)

        print(f"\n🎉 Special topics analysis complete!")
        print(f"📄 Results saved to: {output_file}")
        print(f"💡 Use generate_cluster_report.py to include these in your report")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())