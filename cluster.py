#!/usr/bin/env python3
"""
Clustering script for reduced Bluesky embeddings.
Works with output from reduce.py and performs Gaussian Mixture Model (GMM) clustering + 2D visualization.
Usage: python cluster.py <session_name_dimensions> [options]
"""

import json
import argparse
import numpy as np
from pathlib import Path
import umap
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


def load_reduced_embeddings(session_name: str):
    """Load reduced embeddings from file"""
    datasets_dir = Path('datasets')
    # Auto-append '_reduced' if not present
    if not session_name.endswith('_reduced'):
        session_name = f"{session_name}_reduced"
    reduced_file = datasets_dir / f"{session_name}.jsonl"

    if not reduced_file.exists():
        print(f"❌ Reduced embeddings file not found: {reduced_file}")
        print(f"   Run: python reduce.py {session_name.replace('_reduced', '')}")
        return None, None

    print(f"📖 Loading reduced embeddings from {reduced_file}")

    uris = []
    embeddings = []

    with open(reduced_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                uri = entry.get('uri')
                embedding = entry.get('embedding')

                if uri and embedding:
                    uris.append(uri)
                    embeddings.append(embedding)

                if line_num % 10000 == 0:
                    print(f"  📊 Loaded {line_num:,} embeddings...")

            except json.JSONDecodeError:
                continue

    if not embeddings:
        print("❌ No embeddings found!")
        return None, None

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"✅ Loaded {len(embeddings):,} reduced embeddings")
    print(f"📐 Embedding dimension: {embeddings.shape[1]}")

    return uris, embeddings


def cluster_embeddings(embeddings, n_components=30, covariance_type='full'):
    """Cluster embeddings using Gaussian Mixture Model"""
    print(f"\n🎯 Clustering with GMM...")
    print(f"    n_components: {n_components}")
    print(f"    covariance_type: {covariance_type}")

    clusterer = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        verbose=1
    )

    clusterer.fit(embeddings)
    cluster_labels = clusterer.predict(embeddings)
    cluster_probabilities = clusterer.predict_proba(embeddings)

    # Count clusters
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(cluster_labels == -1)

    print(f"✅ Found {n_clusters} clusters")
    print(f"📊 Noise points: {n_noise:,} ({n_noise/len(cluster_labels)*100:.1f}%)")

    # Cluster size distribution
    for label in sorted(unique_labels):
        if label == -1:
            continue
        count = np.sum(cluster_labels == label)
        print(f"  Cluster {label}: {count:,} posts")

    return cluster_labels, cluster_probabilities


def reduce_to_2d(embeddings):
    """Reduce embeddings to 2D for visualization"""
    print(f"\n📈 Reducing to 2D for visualization...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )

    umap_2d = reducer.fit_transform(embeddings)
    print(f"✅ Reduced to 2D")

    return umap_2d


def save_cluster_results(session_name, uris, cluster_labels, cluster_probabilities, umap_2d):
    """Save cluster results to JSON file"""
    print(f"\n💾 Saving cluster results...")

    datasets_dir = Path('datasets')
    # Remove '_reduced' suffix for output file naming
    base_name = session_name.replace('_reduced', '')
    output_file = datasets_dir / f"{base_name}_clusters.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for uri, cluster_id, probabilities, coords in zip(uris, cluster_labels, cluster_probabilities, umap_2d):
            # Get the probability for the assigned cluster
            cluster_probability = float(probabilities[cluster_id])

            result = {
                'uri': uri,
                'cluster_id': int(cluster_id),
                'cluster_probability': cluster_probability,
                'umap_2d_coords': {
                    'x': float(coords[0]),
                    'y': float(coords[1])
                }
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(uris):,} cluster results to {output_file}")
    return output_file




def main():
    parser = argparse.ArgumentParser(description='GMM clustering for reduced Bluesky embeddings')
    parser.add_argument('session_name', help='Session name (e.g., "august") - will automatically look for session_reduced.jsonl')
    parser.add_argument('--n-components', type=int, default=30,
                       help='Number of mixture components/clusters (default: 30)')
    parser.add_argument('--covariance-type', choices=['full', 'tied', 'diag', 'spherical'], default='full',
                       help='Covariance type (default: full)')

    args = parser.parse_args()

    print(f"🚀 Starting clustering for '{args.session_name}'")
    print("=" * 60)

    try:
        # Step 1: Load reduced embeddings
        uris, embeddings = load_reduced_embeddings(args.session_name)
        if embeddings is None:
            return 1

        # Step 2: Cluster
        cluster_labels, cluster_probabilities = cluster_embeddings(
            embeddings,
            args.n_components,
            args.covariance_type
        )

        # Confirmation before 2D visualization
        print(f"\n⏸️  Press Enter to continue with 2D visualization, or Ctrl+C to stop...")
        input()

        # Step 3: Reduce to 2D
        umap_2d = reduce_to_2d(embeddings)

        # Step 4: Save results
        output_file = save_cluster_results(args.session_name, uris, cluster_labels, cluster_probabilities, umap_2d)

        print(f"\n🎉 Clustering complete!")
        print(f"💾 Results saved to: {output_file}")
        print(f"💡 Use analyze_clusters.py to explore the results")

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