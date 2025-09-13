#!/usr/bin/env python3
"""
Step 3: Filter out spam and adult content
Removes posts with unwanted content labels, processing in-place.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class ContentFilter:
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Set up paths
        self.datasets_dir = Path('datasets')
        run_name = self.config['run_name']
        self.processed_file = self.datasets_dir / f"{run_name}_processed.jsonl"
        
        # Content labels to filter out
        self.spam_labels = ['spam']
        self.adult_labels = ['porn', 'sexual', 'nudity', 'sexual-figurative', 'graphic-media']
    
    def is_spam(self, post: Dict[str, Any]) -> bool:
        """Check if post is labeled as spam"""
        labels = post.get('labels', [])
        return any(label.get('val') in self.spam_labels for label in labels)
    
    def is_adult_content(self, post: Dict[str, Any]) -> bool:
        """Check if post contains adult content"""
        labels = post.get('labels', [])
        return any(label.get('val') in self.adult_labels for label in labels)
    
    def run(self):
        """Filter out spam and adult content in-place"""
        print("Filtering out spam and adult content...")
        
        if not self.processed_file.exists():
            print("ERROR: Processed file not found. Run step 2 first.")
            return False
        
        temp_file = self.datasets_dir / f"{self.config['run_name']}_temp_step3.jsonl"
        
        total_posts = 0
        spam_filtered = 0
        adult_filtered = 0
        clean_posts = 0
        
        try:
            with open(self.processed_file, 'r', encoding='utf-8') as infile:
                with open(temp_file, 'w', encoding='utf-8') as outfile:
                    for line_num, line in enumerate(infile, 1):
                        if line_num % 10000 == 0:
                            print(f"  Processed {line_num:,} posts, kept {clean_posts:,} clean posts...")
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            post = json.loads(line)
                            total_posts += 1
                            
                            # Check if post should be filtered
                            should_filter = False
                            
                            if self.is_spam(post):
                                spam_filtered += 1
                                should_filter = True
                            
                            if self.is_adult_content(post):
                                adult_filtered += 1
                                should_filter = True
                            
                            # Keep post if it's clean
                            if not should_filter:
                                outfile.write(line + '\n')
                                clean_posts += 1
                        
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Skipping malformed JSON on line {line_num}: {e}")
                            continue
            
            # Replace original with filtered version
            os.rename(temp_file, self.processed_file)
            
        except Exception as e:
            print(f"ERROR: Content filtering failed: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False
        
        # Results summary
        if total_posts > 0:
            clean_percentage = (clean_posts / total_posts) * 100
        else:
            clean_percentage = 0
        
        print(f"✓ Content filtering complete!")
        print(f"Total posts processed: {total_posts:,}")
        print(f"Spam posts filtered: {spam_filtered:,}")
        print(f"Adult content filtered: {adult_filtered:,}")
        print(f"Clean posts kept: {clean_posts:,} ({clean_percentage:.1f}%)")
        print(f"Updated file: {self.processed_file}")
        
        if self.processed_file.exists():
            file_size = self.processed_file.stat().st_size / (1024*1024)
            print(f"File size: {file_size:.1f} MB")
        
        return True


def main():
    content_filter = ContentFilter()
    success = content_filter.run()
    if not success:
        exit(1)


if __name__ == "__main__":
    main()