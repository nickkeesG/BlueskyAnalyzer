#!/usr/bin/env python3
"""
Step 2: Filter posts with at least minimum likes
Filters the raw dataset to keep only posts with sufficient engagement.
"""

import json
from pathlib import Path
from typing import Dict, Any


class LikesFilter:
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Set up paths
        self.datasets_dir = Path('datasets')
        run_name = self.config['run_name']
        self.input_file = self.datasets_dir / f"{run_name}_raw_data.jsonl"
        self.output_file = self.datasets_dir / f"{run_name}_processed.jsonl"
        self.min_likes = self.config['filters']['min_likes']
    
    def run(self):
        """Filter posts by minimum likes threshold"""
        print(f"Filtering posts with at least {self.min_likes} likes...")
        
        if not self.input_file.exists():
            print(f"ERROR: Input file not found: {self.input_file}")
            return False
        
        # Check if output already exists
        if self.output_file.exists():
            print(f"Output file already exists: {self.output_file}")
            print("Step 2 already completed, skipping...")
            return True
        
        total_posts = 0
        filtered_posts = 0
        
        with open(self.input_file, 'r', encoding='utf-8') as infile:
            with open(self.output_file, 'w', encoding='utf-8') as outfile:
                for line_num, line in enumerate(infile, 1):
                    if line_num % 10000 == 0:
                        print(f"  Processed {line_num:,} posts, kept {filtered_posts:,}...")
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        post = json.loads(line)
                        total_posts += 1
                        
                        like_count = post.get('likeCount', 0)
                        
                        if like_count >= self.min_likes:
                            outfile.write(line + '\n')
                            filtered_posts += 1
                    
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Skipping malformed JSON on line {line_num}: {e}")
                        continue
        
        # Results summary
        if total_posts > 0:
            percentage = (filtered_posts / total_posts) * 100
        else:
            percentage = 0
        
        print(f"✓ Likes filtering complete!")
        print(f"Total posts processed: {total_posts:,}")
        print(f"Posts with ≥{self.min_likes} likes: {filtered_posts:,} ({percentage:.1f}%)")
        print(f"Output file: {self.output_file}")
        
        if self.output_file.exists():
            file_size = self.output_file.stat().st_size / (1024*1024)
            print(f"File size: {file_size:.1f} MB")
        
        return True


def main():
    filter_obj = LikesFilter()
    success = filter_obj.run()
    if not success:
        exit(1)


if __name__ == "__main__":
    main()