#!/usr/bin/env python3
"""
UMAP dimensionality reduction for Bluesky embeddings.
Usage: python reduce.py <session_name> [options]
"""

import json
import argparse
import numpy as np
from pathlib import Path
import umap
import warnings
warnings.filterwarnings('ignore')


def load_embeddings(session_name: str):
    """Load embeddings from embedded data file"""
    datasets_dir = Path('datasets')
    embedded_file = datasets_dir / f"{session_name}_embedded.jsonl"

    if not embedded_file.exists():
        print(f"❌ Embedded file not found: {embedded_file}")
        print(f"   Run: python embed_content.py {session_name}")
        return None, None

    print(f"📖 Loading embeddings from {embedded_file}")

    uris = []
    embeddings = []

    with open(embedded_file, 'r', encoding='utf-8') as f:
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
    print(f"✅ Loaded {len(embeddings):,} embeddings")
    print(f"📐 Original embedding dimension: {embeddings.shape[1]}")

    return uris, embeddings


def reduce_dimensions(embeddings, n_components=50, n_neighbors=30, min_dist=0.0, metric='cosine'):
    """Reduce embeddings using UMAP"""
    print(f"\n🔄 Reducing dimensions to {n_components}D with UMAP...")
    print(f"    n_components: {n_components}")
    print(f"    n_neighbors: {n_neighbors}")
    print(f"    min_dist: {min_dist}")
    print(f"    metric: {metric}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=True
    )

    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"✅ Reduced to {reduced_embeddings.shape[1]}D")

    return reduced_embeddings


def save_reduced_embeddings(session_name, n_components, uris, reduced_embeddings):
    """Save reduced embeddings to JSON file"""
    print(f"\n💾 Saving reduced embeddings...")

    datasets_dir = Path('datasets')
    output_file = datasets_dir / f"{session_name}_reduced.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for uri, embedding in zip(uris, reduced_embeddings):
            result = {
                'uri': uri,
                'embedding': embedding.tolist()
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(uris):,} reduced embeddings to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='UMAP dimensionality reduction for Bluesky embeddings')
    parser.add_argument('session_name', help='Name of the session (e.g., "july", "august")')
    parser.add_argument('--dimensions', '-d', type=int, default=50,
                       help='Target dimensions for reduction (default: 50)')
    parser.add_argument('--neighbors', '-n', type=int, default=100,
                       help='Number of neighbors for UMAP (default: 100)')
    parser.add_argument('--min-dist', type=float, default=0.0,
                       help='Minimum distance for UMAP (default: 0.0)')
    parser.add_argument('--metric', choices=['cosine', 'euclidean', 'manhattan'], default='cosine',
                       help='Distance metric for UMAP (default: cosine)')

    args = parser.parse_args()

    print(f"🚀 Starting dimensionality reduction for '{args.session_name}'")
    print("=" * 60)

    try:
        # Step 1: Load embeddings
        uris, embeddings = load_embeddings(args.session_name)
        if embeddings is None:
            return 1

        # Step 2: Reduce dimensions
        reduced_embeddings = reduce_dimensions(
            embeddings,
            n_components=args.dimensions,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            metric=args.metric
        )

        # Step 3: Save results
        output_file = save_reduced_embeddings(args.session_name, args.dimensions, uris, reduced_embeddings)

        print(f"\n🎉 Reduction complete!")
        print(f"💾 Results saved to: {output_file}")
        print(f"💡 Now run: python cluster.py {args.session_name}_reduced")

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