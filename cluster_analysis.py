#!/usr/bin/env python3
"""
Clustering Analysis Script for Bluesky Embedded Posts
Usage: python cluster_analysis.py <session_name>
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ClusterAnalyzer:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.datasets_dir = Path('datasets')
        self.data = None
        self.embeddings = None
        self.umap_50d = None
        self.umap_2d = None
        self.clusters = None
        self.cluster_labels = None
        
    def load_data(self) -> bool:
        """Load embedded data and join with processed data for metadata"""
        # Look for embedded data file
        embedded_file = self.datasets_dir / f"{self.session_name}_embedded.jsonl"
        processed_file = self.datasets_dir / f"{self.session_name}_processed.jsonl"

        if not embedded_file.exists():
            print(f"❌ Embedded file not found: {embedded_file}")
            print(f"   Run: python embed_content.py {self.session_name}")
            return False

        if not processed_file.exists():
            print(f"❌ Processed file not found: {processed_file}")
            print(f"   Run: python main_pipeline.py first")
            return False

        print(f"📖 Loading embedded data from {embedded_file}")
        print(f"📖 Loading processed data from {processed_file}")

        # Load embedded data (uri, semantic_content, embedding)
        embedded_data = {}
        embeddings_list = []

        with open(embedded_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    uri = entry.get('uri')
                    embedding = entry.get('embedding')

                    if uri and embedding:
                        embedded_data[uri] = {
                            'semantic_content': entry.get('semantic_content', ''),
                            'embedding': embedding
                        }
                        embeddings_list.append(embedding)

                    if line_num % 10000 == 0:
                        print(f"  📊 Processed {line_num:,} embedded entries...")

                except json.JSONDecodeError:
                    continue

        print(f"✅ Loaded {len(embedded_data):,} embeddings")

        # Load processed data and join with embeddings
        posts = []
        embeddings = []

        with open(processed_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    post = json.loads(line.strip())
                    uri = post.get('uri')

                    # Only include posts that have embeddings
                    if uri and uri in embedded_data:
                        # Add embedding and semantic content to the full post data
                        post['embedding'] = embedded_data[uri]['embedding']
                        post['semantic_content'] = embedded_data[uri]['semantic_content']

                        posts.append(post)
                        embeddings.append(embedded_data[uri]['embedding'])

                    if line_num % 10000 == 0:
                        print(f"  📊 Processed {line_num:,} processed posts...")

                except json.JSONDecodeError:
                    continue

        if not posts:
            print(f"❌ No posts with embeddings found!")
            return False

        self.data = posts
        self.embeddings = np.array(embeddings, dtype=np.float32)

        print(f"✅ Loaded {len(posts):,} posts with embeddings and metadata")
        print(f"📐 Embedding dimension: {self.embeddings.shape[1]}")

        return True
    
    def reduce_dimensions_50d(self) -> np.ndarray:
        """First UMAP reduction to 50 dimensions for clustering"""
        print(f"\n🔄 Reducing dimensions to 50D with UMAP...")
        
        reducer = umap.UMAP(
            n_components=50,
            n_neighbors=30,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        
        self.umap_50d = reducer.fit_transform(self.embeddings)
        print(f"✅ Reduced to {self.umap_50d.shape[1]}D")
        
        return self.umap_50d
    
    def cluster_posts(self, min_cluster_size: int = 50, min_samples: int = 10) -> np.ndarray:
        """Cluster posts using HDBSCAN"""
        print(f"\n🎯 Clustering posts with HDBSCAN...")
        print(f"    min_cluster_size: {min_cluster_size}")
        print(f"    min_samples: {min_samples}")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            metric='euclidean'
        )
        
        self.cluster_labels = clusterer.fit_predict(self.umap_50d)
        
        # Count clusters
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.cluster_labels == -1)
        
        print(f"✅ Found {n_clusters} clusters")
        print(f"📊 Noise points: {n_noise:,} ({n_noise/len(self.cluster_labels)*100:.1f}%)")
        
        # Cluster size distribution
        for label in sorted(unique_labels):
            if label == -1:
                continue
            count = np.sum(self.cluster_labels == label)
            print(f"  Cluster {label}: {count:,} posts")
        
        return self.cluster_labels
    
    def reduce_dimensions_2d(self) -> np.ndarray:
        """Second UMAP reduction to 2D for visualization"""
        print(f"\n📈 Reducing to 2D for visualization...")
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        
        self.umap_2d = reducer.fit_transform(self.umap_50d)
        print(f"✅ Reduced to 2D")
        
        return self.umap_2d
    
    def find_cluster_representatives(self, n_top: int = 5) -> Dict[int, List[Dict]]:
        """Find top representative posts for each cluster"""
        print(f"\n🔍 Finding top {n_top} representative posts per cluster...")
        
        cluster_reps = {}
        unique_labels = np.unique(self.cluster_labels)
        
        for label in sorted(unique_labels):
            if label == -1:  # Skip noise
                continue
            
            # Get posts in this cluster
            cluster_mask = self.cluster_labels == label
            cluster_posts = [self.data[i] for i in np.where(cluster_mask)[0]]
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_50d = self.umap_50d[cluster_mask]
            
            if len(cluster_posts) == 0:
                continue
            
            # Calculate cluster centroid in 50D space (better than raw embeddings)
            centroid = np.mean(cluster_50d, axis=0)
            
            # Find posts closest to centroid
            distances = cosine_distances([centroid], cluster_50d)[0]
            top_indices = np.argsort(distances)[:n_top]
            
            representatives = []
            for idx in top_indices:
                post = cluster_posts[idx]
                representatives.append({
                    'post': post,
                    'distance_to_centroid': distances[idx],
                    'semantic_content': post.get('semantic_content', '')[:200] + '...',
                    'author': post.get('author', {}).get('displayName', 'Unknown'),
                    'likes': post.get('likeCount', 0),
                    'reposts': post.get('repostCount', 0)
                })
            
            cluster_reps[label] = representatives
            print(f"  Cluster {label}: {len(representatives)} representatives")
        
        return cluster_reps
    
    def create_visualization(self, cluster_reps: Dict[int, List[Dict]]) -> str:
        """Create interactive Plotly visualization"""
        print(f"\n🎨 Creating interactive visualization...")
        
        # Prepare data for plotting
        df_data = []
        
        for i, post in enumerate(self.data):
            # Get cluster info
            cluster_id = self.cluster_labels[i]
            cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
            
            # Truncate content for hover
            semantic_content = post.get('semantic_content', '')
            if len(semantic_content) > 300:
                semantic_content = semantic_content[:300] + "..."
            
            df_data.append({
                'x': self.umap_2d[i, 0],
                'y': self.umap_2d[i, 1],
                'cluster': cluster_id,
                'cluster_name': cluster_name,
                'author': post.get('author', {}).get('displayName', 'Unknown'),
                'handle': post.get('author', {}).get('handle', 'unknown'),
                'semantic_content': semantic_content,
                'likes': post.get('likeCount', 0),
                'reposts': post.get('repostCount', 0),
                'replies': post.get('replyCount', 0),
                'uri': post.get('uri', ''),
                'created_at': post.get('record', {}).get('createdAt', '')
            })
        
        df = pd.DataFrame(df_data)
        
        # Create color palette
        unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
        color_map[-1] = '#808080'  # Gray for noise
        
        # Create figure
        fig = go.Figure()
        
        # Add points for each cluster
        for cluster in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster]
            cluster_name = f"Cluster {cluster}" if cluster != -1 else "Noise"
            
            fig.add_trace(go.Scatter(
                x=cluster_df['x'],
                y=cluster_df['y'],
                mode='markers',
                name=cluster_name,
                marker=dict(
                    color=color_map[cluster],
                    size=4,
                    opacity=0.7 if cluster != -1 else 0.3
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (@%{customdata[1]})<br>"
                    "<b>Cluster:</b> %{customdata[2]}<br>"
                    "<b>Engagement:</b> ❤️ %{customdata[3]} | 🔄 %{customdata[4]} | 💬 %{customdata[5]}<br>"
                    "<b>Created:</b> %{customdata[6]}<br>"
                    "<b>Content:</b><br>%{customdata[7]}<br>"
                    "<extra></extra>"
                ),
                customdata=np.column_stack([
                    cluster_df['author'],
                    cluster_df['handle'], 
                    cluster_df['cluster_name'],
                    cluster_df['likes'],
                    cluster_df['reposts'],
                    cluster_df['replies'],
                    cluster_df['created_at'],
                    cluster_df['semantic_content']
                ])
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Bluesky Posts Clustering - {self.session_name.title()}",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            width=1200,
            height=800,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=1.01
            )
        )
        
        # Save visualization
        output_file = f"{self.session_name}_clustering_viz.html"
        fig.write_html(output_file)
        print(f"✅ Saved visualization to {output_file}")
        
        return output_file
    
    def save_clustered_data(self) -> str:
        """Save posts with cluster labels to new JSONL file"""
        print(f"\n💾 Saving clustered data...")
        
        output_file = f"{self.session_name}_with_clusters.jsonl"
        posts_saved = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, post in enumerate(self.data):
                # Add cluster information
                cluster_id = int(self.cluster_labels[i])
                cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
                
                # Create enhanced post with cluster info
                clustered_post = {
                    **post,  # All original data
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name,
                    'umap_2d_coords': {
                        'x': float(self.umap_2d[i, 0]),
                        'y': float(self.umap_2d[i, 1])
                    }
                }
                
                f.write(json.dumps(clustered_post, ensure_ascii=False) + '\n')
                posts_saved += 1
        
        print(f"✅ Saved {posts_saved:,} posts with cluster labels to {output_file}")
        return output_file
    
    def print_cluster_summary(self, cluster_reps: Dict[int, List[Dict]]):
        """Print summary of clusters and their top posts"""
        print(f"\n📋 Cluster Analysis Summary for '{self.session_name}'")
        print("=" * 80)
        
        for cluster_id in sorted(cluster_reps.keys()):
            representatives = cluster_reps[cluster_id]
            cluster_size = np.sum(self.cluster_labels == cluster_id)
            
            print(f"\n🎯 CLUSTER {cluster_id} ({cluster_size:,} posts)")
            print("-" * 50)
            
            for i, rep in enumerate(representatives, 1):
                post = rep['post']
                print(f"{i}. {rep['author']} (❤️ {rep['likes']} | 🔄 {rep['reposts']})")
                print(f"   {rep['semantic_content']}")
                print()


def main():
    parser = argparse.ArgumentParser(description='Cluster analysis for Bluesky embedded posts')
    parser.add_argument('session_name', help='Name of the session (e.g., "july", "august")')
    parser.add_argument('--top-posts', '-t', type=int, default=5, 
                       help='Number of top posts per cluster (default: 5)')
    parser.add_argument('--min-cluster-size', type=int, default=50,
                       help='Minimum posts per cluster (default: 50)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Core point threshold for clustering (default: 10)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting cluster analysis for '{args.session_name}'")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ClusterAnalyzer(args.session_name)
    
    try:
        # Step 1: Load data
        if not analyzer.load_data():
            return 1
        
        # Step 2: Reduce to 50D
        analyzer.reduce_dimensions_50d()
        
        # Step 3: Cluster
        analyzer.cluster_posts(args.min_cluster_size, args.min_samples)
        
        # Step 4: Reduce to 2D
        analyzer.reduce_dimensions_2d()
        
        # Step 5: Find representatives
        cluster_reps = analyzer.find_cluster_representatives(args.top_posts)
        
        # Step 6: Create visualization
        viz_file = analyzer.create_visualization(cluster_reps)
        
        # Step 7: Save clustered data
        clustered_file = analyzer.save_clustered_data()
        
        # Step 8: Print summary
        analyzer.print_cluster_summary(cluster_reps)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Interactive visualization: {viz_file}")
        print(f"💾 Clustered data: {clustered_file}")
        print(f"💡 Open the HTML file in your browser to explore clusters")
        
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