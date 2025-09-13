#!/usr/bin/env python3
"""
Dataset Analysis Script for Bluesky Processed Posts
Analyzes posting patterns, engagement metrics, and temporal trends
Usage: python analyze_dataset.py <session_name>
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class DatasetAnalyzer:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.datasets_dir = Path('datasets')
        self.df = None
        
    def load_data(self) -> bool:
        """Load processed dataset for analysis"""
        # Look for processed data file
        processed_file = self.datasets_dir / f"{self.session_name}_processed.jsonl"
        
        if not processed_file.exists():
            print(f"❌ Processed file not found: {processed_file}")
            return False
        
        print(f"📖 Loading processed data from {processed_file}")
        
        # Load posts
        posts = []
        with open(processed_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    post = json.loads(line.strip())
                    posts.append(post)
                    
                    if line_num % 10000 == 0:
                        print(f"  📊 Processed {line_num:,} posts...")
                        
                except json.JSONDecodeError:
                    continue
        
        if not posts:
            print(f"❌ No posts found in dataset!")
            return False
        
        print(f"✅ Loaded {len(posts):,} posts")
        
        # Convert to DataFrame for analysis
        self.df = pd.json_normalize(posts)
        self.process_timestamps()
        
        return True
    
    def process_timestamps(self):
        """Process timestamps for temporal analysis"""
        print(f"⏰ Processing timestamps...")
        
        # Find timestamp column
        timestamp_col = None
        if 'record.createdAt' in self.df.columns:
            timestamp_col = 'record.createdAt'
        elif 'createdAt' in self.df.columns:
            timestamp_col = 'createdAt'
        else:
            print("⚠️  Warning: No timestamp field found")
            print("Available columns:", list(self.df.columns)[:10])
            return
        
        print(f"📅 Using timestamp column: {timestamp_col}")
        
        # Check some sample values
        sample_timestamps = self.df[timestamp_col].dropna().head(3).tolist()
        print(f"Sample timestamps: {sample_timestamps}")
        
        # Parse timestamps with error handling
        try:
            # Parse with UTC timezone handling
            self.df['created_at'] = pd.to_datetime(self.df[timestamp_col], format='ISO8601', errors='coerce', utc=True)
            
            # Check for failed parsing
            failed_count = self.df['created_at'].isna().sum()
            if failed_count > 0:
                print(f"⚠️  Warning: Failed to parse {failed_count} timestamps")
            
            # Filter out failed parses for further processing
            valid_timestamps = self.df['created_at'].notna()
            if valid_timestamps.sum() == 0:
                print("❌ No valid timestamps found")
                return
            
            print(f"✅ Parsed {valid_timestamps.sum():,} valid timestamps")
            print(f"📊 Timestamp dtype: {self.df['created_at'].dtype}")
            
        except Exception as e:
            print(f"❌ Error parsing timestamps: {e}")
            return
        
        # Extract date components only from valid timestamps
        self.df['date'] = self.df['created_at'].dt.date
        self.df['hour'] = self.df['created_at'].dt.hour
        self.df['day_of_week'] = self.df['created_at'].dt.dayofweek  # 0=Monday
        self.df['day_name'] = self.df['created_at'].dt.day_name()
        
        # Calculate date range
        valid_dates = self.df['created_at'].dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            print(f"📅 Date range: {min_date.date()} to {max_date.date()}")
            print(f"📈 Time span: {(max_date - min_date).days + 1} days")
    
    def analyze_posting_patterns(self) -> dict:
        """Analyze temporal posting patterns"""
        print(f"\n📊 Analyzing posting patterns...")
        
        # Posts per day
        daily_counts = self.df.groupby('date').size().reset_index(name='post_count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        # Posts per hour
        hourly_counts = self.df.groupby('hour').size().reset_index(name='post_count')
        
        # Posts per day of week
        dow_counts = self.df.groupby(['day_of_week', 'day_name']).size().reset_index(name='post_count')
        dow_counts = dow_counts.sort_values('day_of_week')
        
        stats = {
            'total_posts': len(self.df),
            'date_range': {
                'start': daily_counts['date'].min(),
                'end': daily_counts['date'].max(),
                'days': len(daily_counts)
            },
            'daily_stats': {
                'mean': daily_counts['post_count'].mean(),
                'median': daily_counts['post_count'].median(),
                'std': daily_counts['post_count'].std(),
                'min': daily_counts['post_count'].min(),
                'max': daily_counts['post_count'].max()
            }
        }
        
        print(f"  Total posts: {stats['total_posts']:,}")
        print(f"  Average posts/day: {stats['daily_stats']['mean']:.1f}")
        print(f"  Median posts/day: {stats['daily_stats']['median']:.1f}")
        print(f"  Min posts/day: {stats['daily_stats']['min']:,}")
        print(f"  Max posts/day: {stats['daily_stats']['max']:,}")
        
        return {
            'stats': stats,
            'daily_counts': daily_counts,
            'hourly_counts': hourly_counts,
            'dow_counts': dow_counts
        }
    
    def analyze_engagement(self) -> dict:
        """Analyze engagement metrics"""
        print(f"\n💬 Analyzing engagement metrics...")
        
        # Extract engagement columns
        engagement_cols = []
        for col in ['likeCount', 'repostCount', 'replyCount', 'quoteCount']:
            if col in self.df.columns:
                engagement_cols.append(col)
                # Fill NaN values with 0
                self.df[col] = self.df[col].fillna(0).astype(int)
        
        if not engagement_cols:
            print("⚠️  No engagement metrics found")
            return {}
        
        # Calculate engagement stats
        engagement_stats = {}
        for col in engagement_cols:
            stats = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'max': self.df[col].max(),
                'total': self.df[col].sum()
            }
            engagement_stats[col] = stats
            print(f"  {col}: avg={stats['mean']:.1f}, median={stats['median']:.0f}, max={stats['max']:,}")
        
        # Calculate total engagement per post
        if len(engagement_cols) > 1:
            self.df['total_engagement'] = self.df[engagement_cols].sum(axis=1)
            engagement_stats['total_engagement'] = {
                'mean': self.df['total_engagement'].mean(),
                'median': self.df['total_engagement'].median(),
                'std': self.df['total_engagement'].std(),
                'max': self.df['total_engagement'].max()
            }
        
        return engagement_stats
    
    def create_visualizations(self, patterns: dict, engagement: dict) -> str:
        """Create comprehensive visualization dashboard"""
        print(f"\n🎨 Creating visualizations...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Posts per Day', 'Posts by Hour of Day',
                'Posts by Day of Week', 'Engagement Distribution',
                'Top Authors by Total Likes', 'Daily Engagement Trends'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08
        )
        
        daily_counts = patterns['daily_counts']
        hourly_counts = patterns['hourly_counts']
        dow_counts = patterns['dow_counts']
        
        # 1. Posts per day (line chart)
        fig.add_trace(
            go.Scatter(
                x=daily_counts['date'],
                y=daily_counts['post_count'],
                mode='lines+markers',
                name='Posts per Day',
                line=dict(color='#1f77b4')
            ),
            row=1, col=1
        )
        
        # 2. Posts by hour (bar chart)
        fig.add_trace(
            go.Bar(
                x=hourly_counts['hour'],
                y=hourly_counts['post_count'],
                name='Posts by Hour',
                marker_color='#ff7f0e'
            ),
            row=1, col=2
        )
        
        # 3. Posts by day of week (bar chart)
        fig.add_trace(
            go.Bar(
                x=dow_counts['day_name'],
                y=dow_counts['post_count'],
                name='Posts by Day',
                marker_color='#2ca02c'
            ),
            row=2, col=1
        )
        
        # 4. Engagement distribution (if available)
        if engagement and 'likeCount' in engagement:
            # Create histogram of like counts
            likes = self.df[self.df['likeCount'] <= self.df['likeCount'].quantile(0.95)]['likeCount']
            fig.add_trace(
                go.Histogram(
                    x=likes,
                    name='Like Distribution',
                    nbinsx=50,
                    marker_color='#d62728'
                ),
                row=2, col=2
            )
        
        # 5. Top authors by like count
        if 'likeCount' in self.df.columns:
            # Use displayName if available, otherwise fall back to handle or did
            author_col = None
            if 'author.displayName' in self.df.columns:
                # Create a combined author label: use displayName if available, else handle
                df_viz = self.df.copy()
                df_viz['author_label'] = df_viz['author.displayName'].fillna('')
                
                # Fill empty display names with handle if available
                if 'author.handle' in self.df.columns:
                    mask = (df_viz['author_label'] == '') | (df_viz['author_label'].isna())
                    df_viz.loc[mask, 'author_label'] = '@' + df_viz.loc[mask, 'author.handle'].fillna('unknown')
                
                # Group by author_label and sum likes
                top_authors_likes = df_viz.groupby('author_label')['likeCount'].sum().sort_values(ascending=False).head(10).reset_index()
                top_authors_likes.columns = ['author', 'total_likes']
                
                fig.add_trace(
                    go.Bar(
                        y=top_authors_likes['author'][::-1],  # Reverse for better display
                        x=top_authors_likes['total_likes'][::-1],
                        name='Top Authors by Likes',
                        orientation='h',
                        marker_color='#9467bd'
                    ),
                    row=3, col=1
                )
        
        # 6. Daily engagement trends (if available)
        if engagement and 'likeCount' in engagement:
            daily_engagement = self.df.groupby('date')['likeCount'].mean().reset_index()
            daily_engagement['date'] = pd.to_datetime(daily_engagement['date'])
            
            fig.add_trace(
                go.Scatter(
                    x=daily_engagement['date'],
                    y=daily_engagement['likeCount'],
                    mode='lines',
                    name='Avg Likes/Day',
                    line=dict(color='#8c564b')
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Dataset Analysis Dashboard - {self.session_name.title()}",
            title_x=0.5,
            showlegend=False
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        fig.update_xaxes(title_text="Day of Week", row=2, col=1)
        fig.update_xaxes(title_text="Like Count", row=2, col=2)
        fig.update_xaxes(title_text="Total Likes", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Posts", row=1, col=1)
        fig.update_yaxes(title_text="Posts", row=1, col=2)
        fig.update_yaxes(title_text="Posts", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Author", row=3, col=1)
        fig.update_yaxes(title_text="Avg Likes", row=3, col=2)
        
        # Save visualization
        output_file = f"{self.session_name}_dataset_analysis.html"
        fig.write_html(output_file)
        print(f"✅ Saved dashboard to {output_file}")
        
        return output_file
    
    def print_summary(self, patterns: dict, engagement: dict):
        """Print analysis summary"""
        print(f"\n📋 Dataset Analysis Summary - '{self.session_name}'")
        print("=" * 60)
        
        stats = patterns['stats']
        print(f"\n📊 Posting Activity:")
        print(f"  Total posts: {stats['total_posts']:,}")
        print(f"  Date range: {stats['date_range']['start'].date()} to {stats['date_range']['end'].date()}")
        print(f"  Time span: {stats['date_range']['days']} days")
        print(f"  Average posts/day: {stats['daily_stats']['mean']:.1f}")
        print(f"  Peak posting day: {stats['daily_stats']['max']:,} posts")
        
        if engagement:
            print(f"\n💬 Engagement Metrics:")
            for metric, stats in engagement.items():
                if metric != 'total_engagement':
                    metric_name = metric.replace('Count', 's')
                    print(f"  Total {metric_name}: {stats['total']:,}")
                    print(f"  Average {metric_name} per post: {stats['mean']:.1f}")
        
        # Content analysis
        if 'semantic_content' in self.df.columns:
            has_semantic = self.df['semantic_content'].notna().sum()
            print(f"\n📝 Content Analysis:")
            print(f"  Posts with semantic content: {has_semantic:,} ({has_semantic/len(self.df)*100:.1f}%)")
        


def main():
    parser = argparse.ArgumentParser(description='Analyze Bluesky processed dataset')
    parser.add_argument('session_name', help='Name of the session (e.g., "july", "august")')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting dataset analysis for '{args.session_name}'")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.session_name)
    
    try:
        # Step 1: Load data
        if not analyzer.load_data():
            return 1
        
        # Step 2: Analyze posting patterns
        patterns = analyzer.analyze_posting_patterns()
        
        # Step 3: Analyze engagement
        engagement = analyzer.analyze_engagement()
        
        # Step 4: Create visualizations
        viz_file = analyzer.create_visualizations(patterns, engagement)
        
        # Step 5: Print summary
        analyzer.print_summary(patterns, engagement)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Dashboard: {viz_file}")
        print(f"💡 Open the HTML file in your browser to explore the data")
        
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