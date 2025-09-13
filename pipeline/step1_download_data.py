#!/usr/bin/env python3
"""
Step 1: Download raw data from Bluesky API
Downloads posts day by day for each keyword with simplified error handling.
"""

import json
import os
import sys
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Set, Any


class APIError(Exception):
    """Custom exception for API-related errors"""
    pass


class BlueskyDataDownloader:
    def __init__(self):
        self.base_url = "https://bsky.social"
        self.search_endpoint = "/xrpc/app.bsky.feed.searchPosts"
        self.auth_endpoint = "/xrpc/com.atproto.server.createSession"
        self.access_token = None
        self.session = requests.Session()
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Set up paths
        self.datasets_dir = Path('datasets')
        self.datasets_dir.mkdir(exist_ok=True)
    
    def authenticate(self) -> bool:
        """Authenticate with Bluesky API"""
        load_dotenv()
        
        handle = os.getenv('BLUESKY_HANDLE')
        password = os.getenv('BLUESKY_APP_PASSWORD')
        
        if not handle or not password:
            print("ERROR: BLUESKY_HANDLE and BLUESKY_APP_PASSWORD must be set in .env file")
            return False
        
        auth_data = {
            "identifier": handle,
            "password": password
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}{self.auth_endpoint}",
                json=auth_data,
                timeout=30
            )
            response.raise_for_status()
            
            auth_response = response.json()
            self.access_token = auth_response.get('accessJwt')
            
            if self.access_token:
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}'
                })
                print("✓ Authenticated with Bluesky")
                return True
            else:
                print("ERROR: No access token received")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Authentication failed: {e}")
            return False
    
    def search_posts(self, keyword: str, since: str, until: str) -> List[Dict[str, Any]]:
        """Search for posts with API request-level retry logic"""
        params = {
            "q": keyword,
            "limit": 100,
            "sort": "latest", 
            "lang": "en",
            "since": since,
            "until": until
        }
        
        for attempt in range(1, 4):  # 3 attempts
            try:
                response = self.session.get(
                    f"{self.base_url}{self.search_endpoint}",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Basic response validation
                if 'posts' not in data:
                    raise APIError(f"Missing 'posts' field. Got keys: {list(data.keys())}")
                
                # Check for rate limit header
                remaining = response.headers.get('ratelimit-remaining')
                if remaining is None:
                    raise APIError("Missing rate limit header in API response")
                
                remaining = int(remaining)
                
                # Simple rate limit management
                if remaining < 100:
                    print(f"  Rate limit low ({remaining}), sleeping 30s...")
                    time.sleep(30)
                
                return data['posts']
                
            except Exception as e:
                print(f"  ✗ API request attempt {attempt} failed: {e}")
                if attempt == 3:
                    raise  # Re-raise the exception after final attempt
                time.sleep(5)  # Brief delay before retry
    
    def filter_posts_by_date_range(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter posts to only include those within the configured date range"""
        filtered_posts = []
        start_date = datetime.strptime(self.config['date_range']['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(self.config['date_range']['end_date'], '%Y-%m-%d').date()
        
        for post in posts:
            post_timestamp = post.get('record', {}).get('createdAt')
            if post_timestamp:
                try:
                    # Parse timestamp - remove Z and parse with timezone
                    dt = datetime.fromisoformat(post_timestamp.replace('Z', '+00:00'))
                    dt_naive = dt.replace(tzinfo=None)  # Remove timezone info for consistency
                    post_date = dt_naive.date()
                    
                    # Check if it's within the configured date range
                    if start_date <= post_date <= end_date:
                        filtered_posts.append(post)
                except ValueError:
                    continue  # Skip posts with invalid timestamps
        
        return filtered_posts
    
    def download_keyword_for_date(self, keyword: str, target_date: str, seen_uris: Set[str]) -> List[Dict[str, Any]]:
        """Download all posts for a keyword on a specific date using moving window with dynamic sizing"""
        print(f"  Downloading keyword: {keyword}")
        
        start_time = f"{target_date}T00:00:00Z"
        end_time = f"{target_date}T23:59:59Z"
        all_new_posts = []
        
        # Dynamic window sizing - start with 5 minutes
        max_window_duration = timedelta(minutes=5)
        
        while True:
            # Apply window size constraint by adjusting start time if needed
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00')).replace(tzinfo=None)
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00')).replace(tzinfo=None)
            actual_duration = end_dt - start_dt
            
            if actual_duration > max_window_duration:
                # Move start time forward to limit window size
                constrained_start_dt = end_dt - max_window_duration
                constrained_start_time = constrained_start_dt.isoformat() + 'Z'
                window_was_constrained = True
                print(f"    Window constrained: {constrained_start_time} to {end_time} (duration: {max_window_duration})")
            else:
                constrained_start_time = start_time
                window_was_constrained = False
                print(f"    Requesting {constrained_start_time} to {end_time} (duration: {actual_duration})")
            
            posts = self.search_posts(keyword, constrained_start_time, end_time)
            posts = self.filter_posts_by_date_range(posts)
            
            new_posts = []
            duplicate_count = 0
            oldest_timestamp = None
            
            # First pass: track oldest timestamp from ALL posts
            for post in posts:
                post_time = post.get('record', {}).get('createdAt')
                if post_time:
                    try:
                        # Parse timestamp and use as next window boundary
                        dt = datetime.fromisoformat(post_time.replace('Z', '+00:00'))
                        dt_naive = dt.replace(tzinfo=None)
                        timestamp_str = dt_naive.isoformat() + 'Z'
                        
                        if oldest_timestamp is None or timestamp_str < oldest_timestamp:
                            oldest_timestamp = timestamp_str
                    except ValueError:
                        continue  # Skip invalid timestamps
            
            # Second pass: categorize posts as new or duplicate
            for post in posts:
                uri = post.get('uri')
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    new_posts.append(post)
                elif uri:
                    duplicate_count += 1
            
            # Adjust max window duration based on post count
            post_count = len(posts)
            if post_count > 85:
                max_window_duration = timedelta(seconds=int(max_window_duration.total_seconds() * 0.9))
                print(f"    Too many posts ({post_count}), shrinking window to {max_window_duration}")
            elif post_count < 20:
                if post_count == 0:
                    if window_was_constrained:
                        max_window_duration = timedelta(seconds=int(max_window_duration.total_seconds() * 2.0))
                        print(f"    No posts found but window constrained, doubling window to {max_window_duration}")
                        continue  # Skip oldest timestamp logic and try again with larger window
                    else:
                        print(f"    No posts found and window not constrained - done with this keyword")
                        break  # Terminate - move to next keyword
                else:
                    max_window_duration = timedelta(seconds=int(max_window_duration.total_seconds() * 2.0))
                    print(f"    Few posts ({post_count}), doubling window to {max_window_duration}")
            elif post_count < 85:
                max_window_duration = timedelta(seconds=int(max_window_duration.total_seconds() * 1.1))
                print(f"    Good post count ({post_count}), growing window to {max_window_duration}")
            
            # If we reach here, post_count > 0, so we process the posts
            all_new_posts.extend(new_posts)
            if duplicate_count > 0:
                print(f"    Found {len(new_posts)} new posts ({duplicate_count} duplicates)")
            else:
                print(f"    Found {len(new_posts)} new posts")
            
            # Move window backward to oldest post timestamp
            if oldest_timestamp:
                # Ensure end_time is never earlier than current constrained_start_time
                if oldest_timestamp < constrained_start_time:
                    end_time = constrained_start_time
                    print(f"    Oldest timestamp ({oldest_timestamp}) was earlier than constrained start time, using {constrained_start_time}")
                else:
                    end_time = oldest_timestamp
            else:
                raise APIError(f"Found {post_count} posts but no valid oldest timestamp - cannot advance window")
            
        
        return all_new_posts
    
    def download_day(self, target_date: str) -> bool:
        """Download all data for a single day"""
        print(f"\n--- Downloading data for {target_date} ---")
        
        temp_file = self.datasets_dir / f"temp_{target_date}.jsonl"
        seen_uris = set()  # Shared across all keywords for this day
        
        for keyword in self.config['keywords']:
            try:
                posts = self.download_keyword_for_date(keyword, target_date, seen_uris)
                
                # Write posts to temp file
                with open(temp_file, 'a', encoding='utf-8') as f:
                    for post in posts:
                        f.write(json.dumps(post, ensure_ascii=False) + '\n')
                
                print(f"  ✓ {keyword}: {len(posts)} posts")
                print("  " + "-" * 40)  # Visual separator between keywords
                
            except Exception as e:
                print(f"  ✗ {keyword} failed: {e}")
                print(f"❌ Day {target_date} failed, abandoning")
                if temp_file.exists():
                    temp_file.unlink()
                return False
        
        # All keywords succeeded - move temp file to main dataset
        main_file = self.datasets_dir / f"{self.config['run_name']}_raw_data.jsonl"
        if temp_file.exists():
            with open(temp_file, 'r', encoding='utf-8') as temp:
                with open(main_file, 'a', encoding='utf-8') as main:
                    main.write(temp.read())
            temp_file.unlink()
        
        print(f"✓ Day {target_date} completed successfully")
        return True
    
    def find_last_complete_date(self) -> str:
        """Find the last complete date in the existing dataset"""
        main_file = self.datasets_dir / f"{self.config['run_name']}_raw_data.jsonl"
        
        if not main_file.exists():
            return self.config['date_range']['start_date']
        
        # Find latest date in existing data
        latest_date = None
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        post = json.loads(line.strip())
                        post_time = post.get('record', {}).get('createdAt')
                        if post_time:
                            dt = datetime.fromisoformat(post_time.replace('Z', '+00:00'))
                            date = dt.replace(tzinfo=None).date()
                            if latest_date is None or date > latest_date:
                                latest_date = date
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as e:
            print(f"Warning: Could not read existing dataset: {e}")
            return self.config['date_range']['start_date']
        
        if latest_date:
            # Start from the day after the latest complete day
            next_day = latest_date + timedelta(days=1)
            return next_day.strftime('%Y-%m-%d')
        else:
            return self.config['date_range']['start_date']
    
    def run(self):
        """Main download process"""
        print("Starting data download...")
        
        start_date = datetime.strptime(self.find_last_complete_date(), '%Y-%m-%d').date()
        end_date = datetime.strptime(self.config['date_range']['end_date'], '%Y-%m-%d').date()
        
        current_date = start_date
        
        print(f"Date range: {start_date} to {end_date}")
        
        if current_date > end_date:
            print("All dates already downloaded!")
            return
        
        while current_date <= end_date:
            # Fresh authentication for each day
            if not self.authenticate():
                print("❌ Authentication failed")
                sys.exit(1)
            
            success = self.download_day(current_date.strftime('%Y-%m-%d'))
            
            if success:
                current_date += timedelta(days=1)
            else:
                print(f"Retrying day {current_date} from the beginning...")
                # Will retry the same day with fresh auth
        
        dataset_filename = f"{self.config['run_name']}_raw_data.jsonl"
        print(f"\n✓ Download complete! Data saved to {self.datasets_dir / dataset_filename}")


def main():
    downloader = BlueskyDataDownloader()
    downloader.run()


if __name__ == "__main__":
    main()