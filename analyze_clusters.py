#!/usr/bin/env python3
"""
Cluster analysis tool for Bluesky posts.
Shows cluster overview and top posts by likes for specific clusters.
Usage:
  python analyze_clusters.py <session_name>                    # Show overview
  python analyze_clusters.py <session_name> --cluster <id>     # Show top posts for cluster
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import random
import os
from dotenv import load_dotenv
from openai import OpenAI


def load_data(session_name: str):
    """Load processed posts and cluster results"""
    datasets_dir = Path('datasets')
    processed_file = datasets_dir / f"{session_name}_processed.jsonl"
    clusters_file = datasets_dir / f"{session_name}_clusters.jsonl"

    if not processed_file.exists():
        print(f"❌ Processed file not found: {processed_file}")
        return None

    if not clusters_file.exists():
        print(f"❌ Clusters file not found: {clusters_file}")
        print(f"   Run: python cluster.py {session_name}")
        return None

    print(f"📖 Loading data from {processed_file}")
    print(f"📖 Loading clusters from {clusters_file}")

    # Load cluster assignments
    cluster_map = {}
    with open(clusters_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                uri = entry.get('uri')
                cluster_id = entry.get('cluster_id')
                cluster_probability = entry.get('cluster_probability', None)
                if uri and cluster_id is not None:
                    cluster_map[uri] = {
                        'cluster_id': cluster_id,
                        'cluster_probability': cluster_probability
                    }
            except json.JSONDecodeError:
                continue

    # Load posts and add cluster info
    posts = []
    with open(processed_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                post = json.loads(line.strip())
                uri = post.get('uri')
                if uri in cluster_map:
                    post['cluster_id'] = cluster_map[uri]['cluster_id']
                    post['cluster_probability'] = cluster_map[uri]['cluster_probability']
                    posts.append(post)
            except json.JSONDecodeError:
                continue

    print(f"✅ Loaded {len(posts):,} posts with cluster assignments")
    return posts


def show_cluster_overview(posts):
    """Show overview of cluster sizes and average normalized score for top 10 posts"""
    cluster_counts = Counter(post['cluster_id'] for post in posts)

    def get_score(post):
        likes = post.get('likeCount', 0)
        prob = post.get('cluster_probability', 1.0)  # Default to 1.0 if no probability
        return likes * (prob ** 2) if prob is not None else likes

    print(f"\n📊 Cluster Overview")
    print("=" * 80)

    print(f"Clusters: {len(cluster_counts)} total")
    print(f"Total posts: {len(posts):,}")
    print("-" * 80)

    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        # Calculate average normalized score for top 10 posts in this cluster
        cluster_posts = [p for p in posts if p['cluster_id'] == cluster_id]
        cluster_posts.sort(key=get_score, reverse=True)
        top_10_posts = cluster_posts[:10]
        avg_score = sum(get_score(p) for p in top_10_posts) / len(top_10_posts) if top_10_posts else 0
        print(f"Cluster {cluster_id:2d}: {count:,} posts ({count/len(posts)*100:.1f}%) | Avg score (top 10): {avg_score:.1f}")


def show_top_posts(posts, cluster_id, top_n=10):
    """Show top posts by likes*probability for a specific cluster"""
    cluster_posts = [p for p in posts if p['cluster_id'] == cluster_id]

    if not cluster_posts:
        print(f"❌ No posts found for cluster {cluster_id}")
        return

    # Sort by like count * probability (descending)
    def get_score(post):
        likes = post.get('likeCount', 0)
        prob = post.get('cluster_probability', 1.0)  # Default to 1.0 if no probability
        return likes * (prob ** 2) if prob is not None else likes

    cluster_posts.sort(key=get_score, reverse=True)
    top_posts = cluster_posts[:top_n]

    print(f"\n🏆 Top {len(top_posts)} posts in Cluster {cluster_id} (by likes × probability²)")
    print(f"Total posts in cluster {cluster_id}: {len(cluster_posts):,}")
    print("=" * 80)

    for i, post in enumerate(top_posts, 1):
        author = post.get('author', {})
        display_name = author.get('displayName', 'Unknown')
        handle = author.get('handle', 'unknown')

        likes = post.get('likeCount', 0)
        reposts = post.get('repostCount', 0)
        replies = post.get('replyCount', 0)
        cluster_prob = post.get('cluster_probability')

        # Get post text
        record = post.get('record', {})
        text = record.get('text', '')
        if len(text) > 200:
            text = text[:200] + "..."

        created_at = record.get('createdAt', '')
        if created_at:
            # Just show the date part
            created_at = created_at[:10]

        # Convert AT URI to Bluesky URL
        uri = post.get('uri', '')
        bluesky_url = ''
        if uri.startswith('at://'):
            # Extract did and post ID from AT URI
            # Format: at://did:plc:xyz/app.bsky.feed.post/abc123
            parts = uri.replace('at://', '').split('/')
            if len(parts) >= 3:
                did = parts[0]
                post_id = parts[2]
                bluesky_url = f"https://bsky.app/profile/{did}/post/{post_id}"

        # Format probability display
        prob_text = f" | 🎯 {cluster_prob:.1%}" if cluster_prob is not None else ""

        print(f"{i:2d}. {display_name} (@{handle})")
        print(f"    ❤️ {likes:,} | 🔄 {reposts:,} | 💬 {replies:,} | 📅 {created_at}{prob_text}")
        print(f"    {text}")
        if bluesky_url:
            print(f"    🔗 {bluesky_url}")
        print()


def generate_cluster_descriptions(posts, session_name):
    """Generate AI-powered cluster descriptions using GPT-5 mini"""
    print(f"🤖 Generating AI cluster descriptions...")

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("❌ Error: OPENAI_API_KEY must be set in .env file")
        return None

    client = OpenAI(api_key=api_key)

    # Load prompt template
    prompt_file = Path('prompts/cluster_description.txt')
    if not prompt_file.exists():
        print(f"❌ Prompt file not found: {prompt_file}")
        return None

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    def get_score(post):
        likes = post.get('likeCount', 0)
        prob = post.get('cluster_probability', 1.0)
        return likes * (prob ** 2) if prob is not None else likes

    cluster_counts = Counter(post['cluster_id'] for post in posts)

    # Get top 15 posts for each cluster
    all_cluster_top_posts = {}
    for cluster_id in sorted(cluster_counts.keys()):
        cluster_posts = [p for p in posts if p['cluster_id'] == cluster_id]
        cluster_posts.sort(key=get_score, reverse=True)
        top_15_posts = cluster_posts[:15]
        all_cluster_top_posts[cluster_id] = top_15_posts

    # Prepare output
    descriptions = []
    output_dir = Path('datasets')
    output_file = output_dir / f"{session_name}_cluster_descriptions.jsonl"

    print(f"📝 Processing {len(cluster_counts)} clusters...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for cluster_id in sorted(cluster_counts.keys()):
            print(f"  🎯 Analyzing cluster {cluster_id}...")

            # Get target posts
            target_posts = all_cluster_top_posts[cluster_id]
            target_content = []
            for i, post in enumerate(target_posts, 1):
                semantic_content = post.get('semantic_content', '')
                target_content.append(f"{i}. {semantic_content}")

            # Get comparison posts (10 random from other clusters)
            comparison_posts = []
            other_cluster_ids = [cid for cid in all_cluster_top_posts.keys() if cid != cluster_id]

            if other_cluster_ids:
                # Collect all posts from other clusters
                all_other_posts = []
                for other_cid in other_cluster_ids:
                    all_other_posts.extend(all_cluster_top_posts[other_cid])

                # Randomly sample 10 posts
                sampled_posts = random.sample(all_other_posts, min(10, len(all_other_posts)))
                for i, post in enumerate(sampled_posts, 1):
                    semantic_content = post.get('semantic_content', '')
                    comparison_posts.append(f"{chr(64 + i)}. {semantic_content}")

            # Format prompt
            target_posts_text = "\n".join(target_content)
            comparison_posts_text = "\n".join(comparison_posts) if comparison_posts else "No comparison posts available."

            prompt = prompt_template.replace(
                "{target_posts}", target_posts_text
            ).replace(
                "{comparison_posts}", comparison_posts_text
            )

            try:
                # Call GPT-5 mini
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=2000
                )

                response_text = response.choices[0].message.content
                if response_text is None:
                    response_text = ""
                response_text = response_text.strip()

                # Parse JSON response
                try:
                    ai_analysis = json.loads(response_text)
                    theme = ai_analysis.get('theme', 'No theme provided')
                    keywords = ai_analysis.get('keywords', [])
                except json.JSONDecodeError:
                    print(f"    ⚠️  Warning: Invalid JSON response for cluster {cluster_id}")
                    theme = response_text
                    keywords = []

                # Create cluster description entry
                description = {
                    'cluster_id': cluster_id,
                    'post_count': len([p for p in posts if p['cluster_id'] == cluster_id]),
                    'theme': theme,
                    'keywords': keywords
                }

                descriptions.append(description)
                f.write(json.dumps(description, ensure_ascii=False) + '\n')

                print(f"    ✅ Generated: {theme[:50]}...")

            except Exception as e:
                print(f"    ❌ Error calling GPT-5 mini for cluster {cluster_id}: {e}")
                # Create fallback entry
                fallback_description = {
                    'cluster_id': cluster_id,
                    'post_count': len([p for p in posts if p['cluster_id'] == cluster_id]),
                    'theme': 'Error generating description',
                    'keywords': []
                }
                descriptions.append(fallback_description)
                f.write(json.dumps(fallback_description, ensure_ascii=False) + '\n')

    print(f"📄 Cluster descriptions saved to: {output_file}")
    return output_file, descriptions


def classify_cluster_relevance(descriptions, session_name):
    """Classify all clusters as relevant or irrelevant to AI discourse in a single batch call"""
    print(f"🎯 Classifying cluster relevance...")

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("❌ Error: OPENAI_API_KEY must be set in .env file")
        return descriptions

    client = OpenAI(api_key=api_key)

    # Load prompt template
    prompt_file = Path('prompts/batch_relevance.txt')
    if not prompt_file.exists():
        print(f"❌ Batch relevance prompt file not found: {prompt_file}")
        return descriptions

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Format cluster descriptions for the prompt
    cluster_desc_text = []
    for desc in descriptions:
        cluster_id = desc['cluster_id']
        theme = desc['theme']
        keywords = desc.get('keywords', [])
        keywords_str = ', '.join(keywords) if keywords else 'None'
        cluster_desc_text.append(f"Cluster {cluster_id}: {theme} | Keywords: {keywords_str}")

    cluster_descriptions_formatted = "\n".join(cluster_desc_text)

    # Create prompt
    prompt = prompt_template.replace("{cluster_descriptions}", cluster_descriptions_formatted)

    try:
        # Call GPT-5 mini for batch classification
        print(f"  🤖 Calling GPT-5 mini to classify {len(descriptions)} clusters...")
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=2000
        )

        response_text = response.choices[0].message.content
        if response_text is None:
            response_text = ""
        response_text = response_text.strip()

        # Parse JSON response
        try:
            relevance_classifications = json.loads(response_text)

            # Create lookup dict for relevance info
            relevance_lookup = {}
            for classification in relevance_classifications:
                cluster_id = classification.get('cluster_id')
                relevance = classification.get('relevance', 'unknown')
                confidence = classification.get('confidence', 'unknown')
                if cluster_id is not None:
                    relevance_lookup[cluster_id] = {
                        'relevance': relevance,
                        'confidence': confidence
                    }

            # Add relevance info to descriptions
            updated_descriptions = []
            for desc in descriptions:
                cluster_id = desc['cluster_id']
                if cluster_id in relevance_lookup:
                    desc['relevance'] = relevance_lookup[cluster_id]['relevance']
                    desc['confidence'] = relevance_lookup[cluster_id]['confidence']
                else:
                    desc['relevance'] = 'unknown'
                    desc['confidence'] = 'unknown'
                updated_descriptions.append(desc)

            # Count relevance classifications
            relevant_count = sum(1 for d in updated_descriptions if d.get('relevance') == 'relevant')
            irrelevant_count = sum(1 for d in updated_descriptions if d.get('relevance') == 'irrelevant')

            print(f"  ✅ Classification complete: {relevant_count} relevant, {irrelevant_count} irrelevant")

            # Save updated descriptions
            output_dir = Path('datasets')
            output_file = output_dir / f"{session_name}_cluster_descriptions.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for desc in updated_descriptions:
                    f.write(json.dumps(desc, ensure_ascii=False) + '\n')

            print(f"📄 Updated cluster descriptions saved to: {output_file}")
            return updated_descriptions

        except json.JSONDecodeError:
            print(f"    ⚠️  Warning: Invalid JSON response for batch classification")
            print(f"    Raw response: {response_text[:200]}...")
            return descriptions

    except Exception as e:
        print(f"    ❌ Error calling GPT-5 mini for relevance classification: {e}")
        return descriptions


def generate_cluster_report(posts, session_name):
    """Generate a text report with top 6 posts for each cluster"""
    from collections import Counter

    def get_score(post):
        likes = post.get('likeCount', 0)
        prob = post.get('cluster_probability', 1.0)
        return likes * (prob ** 2) if prob is not None else likes

    cluster_counts = Counter(post['cluster_id'] for post in posts)

    # Prepare output file
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)  # Create reports directory if it doesn't exist
    report_file = reports_dir / f"{session_name}_cluster_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"CLUSTER ANALYSIS REPORT - {session_name.upper()}\n")
        f.write("=" * 60 + "\n\n")

        for cluster_id in sorted(cluster_counts.keys()):
            cluster_posts = [p for p in posts if p['cluster_id'] == cluster_id]
            cluster_posts.sort(key=get_score, reverse=True)
            top_6_posts = cluster_posts[:6]

            f.write(f"CLUSTER {cluster_id}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total posts in cluster: {len(cluster_posts):,}\n\n")

            for i, post in enumerate(top_6_posts, 1):
                # Get semantic content
                semantic_content = post.get('semantic_content', '')

                f.write(f"{semantic_content}\n")
                f.write("-" * 40 + "\n")

            f.write("\n" + "=" * 60 + "\n\n")

    print(f"📄 Cluster report saved to: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description='Analyze Bluesky post clusters')
    parser.add_argument('session_name', help='Name of the session (e.g., "july", "august")')
    parser.add_argument('--cluster', '-c', type=int,
                       help='Show top posts for specific cluster ID')
    parser.add_argument('--top', '-t', type=int, default=10,
                       help='Number of top posts to show (default: 10)')
    parser.add_argument('--generate-descriptions', '-g', action='store_true',
                       help='Generate AI-powered cluster descriptions using GPT-5 mini')

    args = parser.parse_args()

    print(f"🔍 Analyzing clusters for '{args.session_name}'")
    print("=" * 50)

    # Load data
    posts = load_data(args.session_name)
    if posts is None:
        return 1

    if args.cluster is not None:
        # Show top posts for specific cluster
        show_top_posts(posts, args.cluster, args.top)
    elif args.generate_descriptions:
        # Generate AI-powered cluster descriptions
        result = generate_cluster_descriptions(posts, args.session_name)
        if result:
            descriptions_file, descriptions = result
            print(f"\n🤖 AI cluster descriptions generated!")

            # Classify relevance
            updated_descriptions = classify_cluster_relevance(descriptions, args.session_name)

            print(f"📄 Final results saved to: {descriptions_file}")

            # Show brief summary with relevance
            print(f"\n📊 Summary:")
            relevant_clusters = [d for d in updated_descriptions if d.get('relevance') == 'relevant']
            irrelevant_clusters = [d for d in updated_descriptions if d.get('relevance') == 'irrelevant']

            print(f"\n🎯 Relevant Clusters ({len(relevant_clusters)}):")
            for desc in relevant_clusters[:3]:  # Show first 3 relevant
                cluster_id = desc['cluster_id']
                theme = desc['theme']
                post_count = desc['post_count']
                confidence = desc.get('confidence', '')
                print(f"  Cluster {cluster_id} ({post_count:,} posts, {confidence}): {theme[:60]}...")

            if len(irrelevant_clusters) > 0:
                print(f"\n🚫 Irrelevant Clusters ({len(irrelevant_clusters)}):")
                for desc in irrelevant_clusters[:2]:  # Show first 2 irrelevant
                    cluster_id = desc['cluster_id']
                    theme = desc['theme']
                    post_count = desc['post_count']
                    print(f"  Cluster {cluster_id} ({post_count:,} posts): {theme[:60]}...")
    else:
        # Show overview
        show_cluster_overview(posts)

        # Generate cluster report
        report_file = generate_cluster_report(posts, args.session_name)

        print(f"\n💡 Use --cluster <id> to see top posts for a specific cluster")
        print(f"   Example: python analyze_clusters.py {args.session_name} --cluster 0")
        print(f"\n💡 Use --generate-descriptions to create AI-powered cluster descriptions")
        print(f"   Example: python analyze_clusters.py {args.session_name} --generate-descriptions")
        print(f"\n📄 Detailed cluster report generated: {report_file.name}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())