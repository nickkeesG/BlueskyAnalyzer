#!/usr/bin/env python3
"""
Sub-clustering script for Bluesky posts within existing clusters.
Takes posts from each cluster and splits them into 3 subclusters using the same GMM algorithm.
Usage: python sub_cluster.py <session_name>
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_cluster_data(session_name: str):
    """Load cluster assignments from _clusters.jsonl file"""
    datasets_dir = Path('datasets')
    clusters_file = datasets_dir / f"{session_name}_clusters.jsonl"

    if not clusters_file.exists():
        print(f"❌ Clusters file not found: {clusters_file}")
        return None

    print(f"📖 Loading cluster data from {clusters_file}")

    cluster_data = {}
    with open(clusters_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                uri = entry.get('uri')
                cluster_id = entry.get('cluster_id')

                if uri is not None and cluster_id is not None:
                    cluster_data[uri] = entry

                if line_num % 10000 == 0:
                    print(f"  📊 Loaded {line_num:,} cluster assignments...")

            except json.JSONDecodeError:
                continue

    print(f"✅ Loaded {len(cluster_data):,} cluster assignments")
    return cluster_data


def load_embeddings(session_name: str):
    """Load reduced embeddings from _reduced.jsonl file"""
    datasets_dir = Path('datasets')
    reduced_file = datasets_dir / f"{session_name}_reduced.jsonl"

    if not reduced_file.exists():
        print(f"❌ Reduced embeddings file not found: {reduced_file}")
        return None

    print(f"📖 Loading embeddings from {reduced_file}")

    embeddings_data = {}
    with open(reduced_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                uri = entry.get('uri')
                embedding = entry.get('embedding')

                if uri and embedding:
                    embeddings_data[uri] = np.array(embedding, dtype=np.float32)

                if line_num % 10000 == 0:
                    print(f"  📊 Loaded {line_num:,} embeddings...")

            except json.JSONDecodeError:
                continue

    print(f"✅ Loaded {len(embeddings_data):,} embeddings")
    return embeddings_data


def group_by_clusters(cluster_data, embeddings_data):
    """Group posts by their cluster_id"""
    print(f"\n🗂️  Grouping posts by clusters...")

    clusters = defaultdict(list)

    for uri, cluster_info in cluster_data.items():
        if uri in embeddings_data:
            cluster_id = cluster_info['cluster_id']
            clusters[cluster_id].append({
                'uri': uri,
                'embedding': embeddings_data[uri],
                'cluster_info': cluster_info
            })

    # Sort cluster IDs and show summary
    sorted_cluster_ids = sorted(clusters.keys())
    print(f"✅ Found {len(sorted_cluster_ids)} clusters")

    for cluster_id in sorted_cluster_ids:
        count = len(clusters[cluster_id])
        print(f"  Cluster {cluster_id}: {count:,} posts")

    return clusters


def subcluster_posts(posts, cluster_id, n_subclusters=3):
    """Apply GMM subclustering to posts within a cluster"""
    if len(posts) < n_subclusters:
        print(f"  ⚠️  Cluster {cluster_id} has only {len(posts)} posts, assigning all to subcluster 0")
        for post in posts:
            post['subcluster_id'] = 0
        return posts

    # Extract embeddings
    embeddings = np.array([post['embedding'] for post in posts])

    # Apply K-means clustering
    clusterer = KMeans(
        n_clusters=n_subclusters,
        random_state=42,
        n_init=10
    )

    subcluster_labels = clusterer.fit_predict(embeddings)

    # Assign subcluster IDs to posts
    for i, post in enumerate(posts):
        post['subcluster_id'] = int(subcluster_labels[i])

    # Count subclusters
    subcluster_counts = {}
    for label in range(n_subclusters):
        count = np.sum(subcluster_labels == label)
        subcluster_counts[label] = count

    print(f"  Cluster {cluster_id} subclusters: ", end="")
    print(", ".join([f"{label}:{count}" for label, count in sorted(subcluster_counts.items())]))

    return posts


def perform_subclustering(clusters, n_subclusters=3):
    """Perform subclustering on all clusters using K-means"""
    print(f"\n🎯 Performing K-means subclustering with {n_subclusters} subclusters per cluster...")

    all_subclustered_posts = []

    for cluster_id in sorted(clusters.keys()):
        posts = clusters[cluster_id]
        print(f"\n  Processing cluster {cluster_id} ({len(posts):,} posts)...")

        subclustered_posts = subcluster_posts(posts, cluster_id, n_subclusters)
        all_subclustered_posts.extend(subclustered_posts)

    print(f"\n✅ Subclustered {len(all_subclustered_posts):,} posts total")
    return all_subclustered_posts


def save_subclustered_results(session_name, subclustered_posts):
    """Save subclustered results back to the original clusters file"""
    print(f"\n💾 Updating original clusters file with subcluster data...")

    datasets_dir = Path('datasets')
    clusters_file = datasets_dir / f"{session_name}_clusters.jsonl"

    # Create a mapping from URI to subcluster_id for quick lookup
    uri_to_subcluster = {post['uri']: post['subcluster_id'] for post in subclustered_posts}

    # Read the original file and update with subcluster_id
    updated_entries = []
    with open(clusters_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                uri = entry.get('uri')

                if uri in uri_to_subcluster:
                    entry['subcluster_id'] = uri_to_subcluster[uri]

                updated_entries.append(entry)

                if line_num % 10000 == 0:
                    print(f"  📊 Updated {line_num:,} entries...")

            except json.JSONDecodeError:
                continue

    # Write back to the same file
    with open(clusters_file, 'w', encoding='utf-8') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ Updated {len(updated_entries):,} entries in {clusters_file}")
    return clusters_file


def main():
    parser = argparse.ArgumentParser(description='Sub-cluster posts within existing clusters using K-means')
    parser.add_argument('session_name', help='Session name (e.g., "august")')
    parser.add_argument('--n-subclusters', type=int, default=3,
                       help='Number of subclusters per cluster (default: 3)')

    args = parser.parse_args()

    print(f"🚀 Starting subclustering for '{args.session_name}'")
    print("=" * 60)

    try:
        # Step 1: Load cluster assignments
        cluster_data = load_cluster_data(args.session_name)
        if cluster_data is None:
            return 1

        # Step 2: Load embeddings
        embeddings_data = load_embeddings(args.session_name)
        if embeddings_data is None:
            return 1

        # Step 3: Group posts by clusters
        clusters = group_by_clusters(cluster_data, embeddings_data)

        # Step 4: Perform subclustering
        subclustered_posts = perform_subclustering(clusters, args.n_subclusters)

        # Step 5: Save results
        output_file = save_subclustered_results(args.session_name, subclustered_posts)

        print(f"\n🎉 Subclustering complete!")
        print(f"💾 Updated original file: {output_file}")
        print(f"💡 Each post now has a subcluster_id (0-{args.n_subclusters-1}) in addition to its cluster_id")

        return 0

    except KeyboardInterrupt:
        print(f"\n👋 Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())