#!/usr/bin/env python3
"""
Step 4: Enrich replies with parent and root post text
Adds parentText and rootText fields to reply posts for better context.
"""

import json
import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Set, Any


class APIError(Exception):
    """Custom exception for API-related errors"""
    pass


class ReplyEnricher:
    def __init__(self):
        self.base_url = "https://bsky.social"
        self.get_record_endpoint = "/xrpc/com.atproto.repo.getRecord"
        self.auth_endpoint = "/xrpc/com.atproto.server.createSession"
        self.access_token = None
        self.session = requests.Session()
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Set up paths
        self.datasets_dir = Path('datasets')
        run_name = self.config['run_name']
        self.processed_file = self.datasets_dir / f"{run_name}_processed.jsonl"
    
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
    
    def parse_at_uri(self, uri: str) -> Dict[str, str]:
        """Parse AT protocol URI into components"""
        if not uri.startswith('at://'):
            raise ValueError(f"Invalid AT URI: {uri}")
        
        parts = uri[5:].split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid AT URI format: {uri}")
        
        return {
            'repo': parts[0],
            'collection': parts[1],
            'rkey': parts[2]
        }
    
    def fetch_post_by_uri(self, uri: str) -> tuple[str, str]:
        """Fetch a single post by AT URI and return its text"""
        try:
            parsed = self.parse_at_uri(uri)
        except ValueError as e:
            raise APIError(f"Invalid URI format: {e}")
        
        params = {
            'repo': parsed['repo'],
            'collection': parsed['collection'],
            'rkey': parsed['rkey']
        }
        
        for attempt in range(3):
            try:
                response = self.session.get(
                    f"{self.base_url}{self.get_record_endpoint}",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Basic response validation
                if 'value' not in data:
                    raise APIError(f"Missing 'value' field in response for {uri}")
                
                post_data = data['value']
                text = post_data.get('text', '')
                
                # Get rate limit info
                remaining = response.headers.get('ratelimit-remaining', 'unknown')
                
                return text, remaining
                
            except requests.exceptions.RequestException as e:
                # Check if this is a 400 Bad Request for missing/deleted post
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                    error_msg = str(e)
                    if "Bad Request for url" in error_msg:
                        print(f"  Post appears to be deleted/missing (400 Bad Request), treating as empty text")
                        return "", "unknown"  # Return empty string for missing posts - no need to retry
                
                print(f"  Attempt {attempt + 1}/3 failed for {uri}: {e}")
                if attempt == 2:  # Last attempt
                    raise APIError(f"Failed to fetch {uri} after 3 attempts: {e}")
                # Wait before next attempt
                time.sleep(2)
        
        # Should never reach here due to the raise above
        raise APIError(f"Unexpected error fetching {uri}")
    
    def collect_missing_uris(self) -> Set[str]:
        """Collect all parent/root URIs that are still missing"""
        missing_uris = set()
        
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    print(f"  Scanned {line_num:,} posts...")
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    post = json.loads(line)
                    reply = post.get('record', {}).get('reply', {})
                    
                    if reply:
                        # Check if parent text is missing
                        if 'parentText' not in post:
                            parent_uri = reply.get('parent', {}).get('uri')
                            if parent_uri:
                                missing_uris.add(parent_uri)
                        
                        # Check if root text is missing
                        if 'rootText' not in post:
                            root_uri = reply.get('root', {}).get('uri')
                            if root_uri:
                                missing_uris.add(root_uri)
                
                except json.JSONDecodeError:
                    continue
        
        return missing_uris
    
    def update_posts_with_batch(self, fetched_texts: Dict[str, str]):
        """Update all posts that need URIs from the fetched batch"""
        temp_file = self.datasets_dir / f"{self.config['run_name']}_temp_step4.jsonl"
        updated_count = 0
        
        try:
            with open(self.processed_file, 'r', encoding='utf-8') as infile:
                with open(temp_file, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            post = json.loads(line)
                            reply = post.get('record', {}).get('reply', {})
                            
                            if reply:
                                # Add parent text if URI is in batch and field is missing
                                parent_uri = reply.get('parent', {}).get('uri')
                                if parent_uri in fetched_texts and 'parentText' not in post:
                                    post['parentText'] = fetched_texts[parent_uri]
                                    updated_count += 1
                                
                                # Add root text if URI is in batch and field is missing
                                root_uri = reply.get('root', {}).get('uri')
                                if root_uri in fetched_texts and 'rootText' not in post:
                                    post['rootText'] = fetched_texts[root_uri]
                                    updated_count += 1
                            
                            outfile.write(json.dumps(post, ensure_ascii=False) + '\n')
                            
                        except json.JSONDecodeError:
                            # Keep malformed lines as-is
                            outfile.write(line + '\n')
                            continue
            
            # Replace original with updated version
            os.rename(temp_file, self.processed_file)
            print(f"  Updated {updated_count} posts with batch of {len(fetched_texts)} URIs")
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise APIError(f"Failed to update dataset: {e}")
    
    def run(self):
        """Main enrichment process"""
        print("Enriching replies with parent/root context...")
        
        if not self.processed_file.exists():
            print("ERROR: Processed file not found. Run steps 1-3 first.")
            return False
        
        if not self.authenticate():
            print("ERROR: Authentication failed")
            return False
        
        while True:
            # Find what's missing
            print("Scanning dataset for missing parent/root URIs...")
            missing_uris = self.collect_missing_uris()
            
            if not missing_uris:
                print("✓ All replies already enriched!")
                return True
            
            print(f"Found {len(missing_uris)} URIs still need fetching")
            
            # Take batch of up to 500 URIs
            batch = list(missing_uris)[:500]
            print(f"Processing batch of {len(batch)} URIs...")
            
            try:
                # Fetch all URIs in the batch
                fetched_texts = {}
                last_rate_limit = None
                for i, uri in enumerate(batch, 1):
                    text, rate_limit = self.fetch_post_by_uri(uri)  # 3 attempts, raises APIError if fails
                    fetched_texts[uri] = text
                    last_rate_limit = rate_limit

                print(f"  Fetched {len(batch)} URIs, final rate limit: {last_rate_limit}")
                
                # Update dataset with entire batch at once
                self.update_posts_with_batch(fetched_texts)
                
            except APIError as e:
                print(f"Batch failed: {e}")
                print("Restarting step 4 in 10 seconds...")
                time.sleep(10)
                return self.run()  # Restart entire step


def main():
    enricher = ReplyEnricher()
    success = enricher.run()
    if not success:
        exit(1)


if __name__ == "__main__":
    main()