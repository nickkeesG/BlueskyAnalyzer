#!/usr/bin/env python3
"""
Embedding Content Script for Bluesky Datasets
Takes a run name and embeds all semantic content, waiting for completion.
Creates both embedded.jsonl (with embeddings) and with_embeddings.jsonl (processed data + embeddings).
"""

import json
import os
import sys
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path to import existing batch system
sys.path.append(str(Path(__file__).parent.parent))
from openai import OpenAI


class EmbeddingProcessor:
    def __init__(self, api_key: str, chunk_size: int = 1000):
        self.client = OpenAI(api_key=api_key)
        self.chunk_size = chunk_size
        self.datasets_dir = Path('datasets')
        self.temp_dir = None
        
    def validate_input(self, run_name: str, force: bool = False) -> Path:
        """Validate input file and check for existing outputs"""
        processed_file = self.datasets_dir / f"{run_name}_processed.jsonl"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"No input file found for run '{run_name}'. Looked for:\n"
                                  f"  - {processed_file}")
        
        print(f"✅ Found processed file: {processed_file}")
        
        # Check for existing outputs
        embedded_file = self.datasets_dir / f"{run_name}_embedded.jsonl"

        if not force and embedded_file.exists():
            print(f"⚠️  Output file already exists:")
            print(f"  - {embedded_file}")
            response = input("Continue anyway? This will overwrite existing file. [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Aborted.")
                sys.exit(0)
        
        return processed_file
    
    def count_posts_with_content(self, input_file: Path) -> int:
        """Count posts that have semantic content for embedding"""
        count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    post = json.loads(line.strip())
                    semantic_content = post.get('semantic_content', '').strip()
                    if semantic_content:
                        count += 1
                except json.JSONDecodeError:
                    continue
        return count
    
    def prepare_batch_chunks(self, input_file: Path, run_name: str) -> Dict:
        """Prepare batch files by chunking the input"""
        print(f"📦 Preparing batch chunks for embedding...")
        
        # Create temporary directory for batch files
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"embed_{run_name}_"))
        print(f"📁 Working directory: {self.temp_dir}")
        
        # Read posts with semantic content
        posts_with_content = []
        posts_total = 0
        posts_without_content = 0
        
        print(f"📖 Reading posts from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    post = json.loads(line.strip())
                    posts_total += 1
                    
                    semantic_content = post.get('semantic_content', '').strip()
                    if semantic_content:
                        post['_line_number'] = line_num
                        posts_with_content.append(post)
                    else:
                        posts_without_content += 1
                        
                    if posts_total % 5000 == 0:
                        print(f"  📊 Read {posts_total:,} posts...")
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"📊 Dataset summary:")
        print(f"  Total posts: {posts_total:,}")
        print(f"  Posts with semantic content: {len(posts_with_content):,}")
        print(f"  Posts without content: {posts_without_content:,}")
        
        if not posts_with_content:
            raise ValueError("No posts with semantic content found!")
        
        # Calculate chunks
        num_chunks = (len(posts_with_content) + self.chunk_size - 1) // self.chunk_size
        print(f"📦 Creating {num_chunks} chunks of up to {self.chunk_size} posts each")
        
        # Create chunk files and mappings
        chunk_info = []
        global_mapping = {}
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(posts_with_content))
            chunk_posts = posts_with_content[start_idx:end_idx]
            
            chunk_name = f"{run_name}_chunk_{chunk_idx + 1:03d}"
            batch_file = self.temp_dir / f"{chunk_name}.jsonl"
            
            print(f"  📄 Creating {chunk_name}: {len(chunk_posts)} posts")
            
            # Create batch requests
            with open(batch_file, 'w', encoding='utf-8') as f:
                for post_idx, post in enumerate(chunk_posts):
                    custom_id = f"chunk_{chunk_idx + 1:03d}_post_{post_idx + 1:04d}"
                    post_uri = post.get('uri', f'line_{post.get("_line_number", start_idx + post_idx + 1)}')
                    
                    # Store mapping
                    global_mapping[post_uri] = {
                        'chunk': chunk_idx + 1,
                        'custom_id': custom_id,
                        'line_number': post.get('_line_number'),
                        'semantic_content': post['semantic_content']  # Store for embedded.jsonl creation
                    }
                    
                    # Create batch API request
                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST", 
                        "url": "/v1/embeddings",
                        "body": {
                            "model": "text-embedding-3-large",
                            "input": post['semantic_content'],
                            "encoding_format": "float"
                        }
                    }
                    
                    f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')
            
            chunk_info.append({
                'chunk_id': chunk_idx + 1,
                'chunk_name': chunk_name,
                'batch_file': str(batch_file),
                'post_count': len(chunk_posts)
            })
        
        # Save global mapping
        global_mapping_file = self.temp_dir / f"{run_name}_global_mapping.json"
        with open(global_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(global_mapping, f, indent=2, ensure_ascii=False)
        
        return {
            'run_name': run_name,
            'input_file': str(input_file),
            'total_posts': len(posts_with_content),
            'num_chunks': num_chunks,
            'chunks': chunk_info,
            'global_mapping_file': str(global_mapping_file),
            'temp_dir': str(self.temp_dir)
        }
    
    def upload_and_create_batches(self, batch_data: Dict) -> List[Dict]:
        """Upload batch files and create OpenAI batch jobs"""
        print(f"\n🚀 Starting batch upload and job creation...")
        
        batch_jobs = []
        chunks = batch_data['chunks']
        
        for i, chunk_info in enumerate(chunks, 1):
            chunk_name = chunk_info['chunk_name']
            batch_file = chunk_info['batch_file']
            
            print(f"[{i}/{len(chunks)}] Processing {chunk_name}...")
            
            try:
                # Upload batch file
                print(f"  📤 Uploading...")
                with open(batch_file, 'rb') as f:
                    batch_input_file = self.client.files.create(
                        file=f,
                        purpose="batch"
                    )
                
                file_id = batch_input_file.id
                print(f"  ✅ Uploaded: {file_id}")
                
                # Create batch job
                print(f"  🔄 Creating batch job...")
                batch = self.client.batches.create(
                    input_file_id=file_id,
                    endpoint="/v1/embeddings",
                    completion_window="24h",
                    metadata={"description": f"Embeddings for {batch_data['run_name']} - {chunk_name}"}
                )
                
                batch_id = batch.id
                print(f"  ✅ Created: {batch_id}")
                
                batch_jobs.append({
                    'batch_id': batch_id,
                    'file_id': file_id,
                    'chunk_info': chunk_info,
                    'status': batch.status,
                    'created_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                raise
        
        print(f"\n✅ All {len(batch_jobs)} batch jobs created successfully!")
        return batch_jobs
    
    def download_single_chunk(self, job: Dict) -> Optional[Path]:
        """Download results for a single completed chunk"""
        batch_id = job['batch_id']
        chunk_name = job['chunk_info']['chunk_name']
        output_file_id = job.get('output_file_id')
        
        if not output_file_id:
            return None
        
        output_file = self.temp_dir / f"{chunk_name}_results.jsonl"
        
        try:
            print(f"    📥 Downloading {chunk_name}...")
            result = self.client.files.content(output_file_id)
            with open(output_file, 'wb') as f:
                f.write(result.content)
            
            # Count successful embeddings
            embedding_count = 0
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        result_data = json.loads(line.strip())
                        response = result_data.get('response', {})
                        if response.get('status_code') == 200:
                            embedding_count += 1
                    except json.JSONDecodeError:
                        continue
            
            file_size = output_file.stat().st_size
            print(f"    ✅ Downloaded: {embedding_count} embeddings ({file_size:,} bytes)")
            job['downloaded'] = True
            job['embedding_count'] = embedding_count
            return output_file
            
        except Exception as e:
            print(f"    ❌ Download error: {e}")
            return None

    def monitor_batches(self, batch_jobs: List[Dict], check_interval: int) -> bool:
        """Monitor batch jobs until completion, downloading results immediately"""
        print(f"\n👀 Monitoring {len(batch_jobs)} batch jobs...")
        print(f"⏰ Checking status every {check_interval // 60} minutes")
        
        completed_batches = set()
        downloaded_files = []
        
        while len(completed_batches) < len(batch_jobs):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking batch status...")
            
            status_counts = {'validating': 0, 'in_progress': 0, 'finalizing': 0, 'completed': 0, 'failed': 0}
            total_posts = 0
            completed_posts = 0
            
            for job in batch_jobs:
                batch_id = job['batch_id']
                chunk_name = job['chunk_info']['chunk_name']
                post_count = job['chunk_info']['post_count']
                total_posts += post_count
                
                try:
                    batch = self.client.batches.retrieve(batch_id)
                    status = batch.status
                    status_counts[status] = status_counts.get(status, 0) + 1
                    
                    # Print status for every batch
                    if status == 'completed':
                        if batch_id not in completed_batches:
                            # Newly completed - download immediately
                            completed_batches.add(batch_id)
                            job['output_file_id'] = batch.output_file_id
                            print(f"  ✅ {chunk_name}: COMPLETED ({post_count} posts)")
                            
                            # Download immediately
                            downloaded_file = self.download_single_chunk(job)
                            if downloaded_file:
                                downloaded_files.append(downloaded_file)
                        else:
                            # Already downloaded
                            embedding_count = job.get('embedding_count', post_count)
                            print(f"  📥 {chunk_name}: COMPLETED & DOWNLOADED ({embedding_count} embeddings)")
                        completed_posts += post_count
                    elif status == 'failed':
                        print(f"  ❌ {chunk_name}: FAILED")
                        completed_batches.add(batch_id)  # Don't wait for failed batches
                    elif status == 'validating':
                        print(f"  🔄 {chunk_name}: VALIDATING ({post_count} posts)")
                    elif status == 'in_progress':
                        req_counts = batch.request_counts
                        progress = f"({req_counts.completed}/{req_counts.total})"
                        print(f"  ⚡ {chunk_name}: IN_PROGRESS {progress}")
                        completed_posts += req_counts.completed
                    elif status == 'finalizing':
                        req_counts = batch.request_counts
                        progress = f"({req_counts.completed}/{req_counts.total})"
                        print(f"  🔧 {chunk_name}: FINALIZING {progress}")
                        completed_posts += req_counts.completed
                    else:
                        print(f"  ❓ {chunk_name}: {status.upper()} ({post_count} posts)")
                    
                except Exception as e:
                    print(f"  ❌ Error checking {chunk_name}: {e}")
            
            # Print summary
            print(f"\n📊 Progress Summary:")
            print(f"  ✅ Completed: {status_counts['completed']}/{len(batch_jobs)} chunks")
            print(f"  📥 Downloaded: {len(downloaded_files)} chunks")
            print(f"  ⚡ In Progress: {status_counts['in_progress']} chunks")
            print(f"  🔄 Validating: {status_counts['validating']} chunks") 
            print(f"  🔧 Finalizing: {status_counts['finalizing']} chunks")
            if status_counts['failed'] > 0:
                print(f"  ❌ Failed: {status_counts['failed']} chunks")
            
            completion_pct = (completed_posts / total_posts) * 100 if total_posts > 0 else 0
            print(f"  📈 Overall Progress: {completed_posts:,}/{total_posts:,} posts ({completion_pct:.1f}%)")
            
            if len(completed_batches) >= len(batch_jobs):
                print(f"\n🎉 All batches completed!")
                break
            
            print(f"  ⏰ Next check in {check_interval // 60} minutes...")
            time.sleep(check_interval)
        
        # Store downloaded files for later use
        self._downloaded_files = downloaded_files
        
        # Check for failures
        failed_count = status_counts.get('failed', 0)
        if failed_count > 0:
            print(f"\n⚠️  Warning: {failed_count} chunks failed. Results will be partial.")
            return False
        
        return True
    
    def download_results(self, batch_jobs: List[Dict]) -> List[Path]:
        """Return already downloaded files from monitoring process"""
        if hasattr(self, '_downloaded_files'):
            print(f"\n✅ Using {len(self._downloaded_files)} files downloaded during monitoring")
            return self._downloaded_files
        
        # Fallback: download any remaining files (shouldn't happen with new approach)
        print(f"\n📥 Downloading any remaining batch results...")
        downloaded_files = []
        
        for job in batch_jobs:
            if job.get('downloaded'):
                continue  # Already downloaded during monitoring
            
            batch_id = job['batch_id']
            chunk_name = job['chunk_info']['chunk_name'] 
            output_file_id = job.get('output_file_id')
            
            if not output_file_id:
                print(f"  ⏭️  Skipping {chunk_name} - no output file")
                continue
            
            print(f"  📄 Downloading remaining {chunk_name}...")
            downloaded_file = self.download_single_chunk(job)
            if downloaded_file:
                downloaded_files.append(downloaded_file)
        
        if downloaded_files:
            print(f"✅ Downloaded {len(downloaded_files)} additional result files")
        
        return getattr(self, '_downloaded_files', []) + downloaded_files
    
    def create_output_files(self, batch_data: Dict, downloaded_files: List[Path]) -> Path:
        """Create embedded.jsonl file"""
        print(f"\n🔗 Creating output files...")
        
        run_name = batch_data['run_name']
        input_file = Path(batch_data['input_file'])
        global_mapping_file = Path(batch_data['global_mapping_file'])
        
        # Load global mapping
        with open(global_mapping_file, 'r') as f:
            global_mapping = json.load(f)
        
        # Load all embeddings from downloaded files
        embeddings = {}
        print(f"📖 Loading embeddings from {len(downloaded_files)} files...")
        
        for result_file in downloaded_files:
            print(f"  📄 Processing {result_file.name}...")
            
            with open(result_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        custom_id = result.get('custom_id')
                        
                        if not custom_id:
                            continue
                        
                        # Check if request succeeded
                        response = result.get('response', {})
                        if response.get('status_code') != 200:
                            continue
                        
                        # Extract embedding
                        body = response.get('body', {})
                        data = body.get('data', [])
                        if not data:
                            continue
                        
                        embedding = data[0].get('embedding')
                        if not embedding:
                            continue
                        
                        # Find original URI
                        post_uri = None
                        for uri, mapping_data in global_mapping.items():
                            if mapping_data['custom_id'] == custom_id:
                                post_uri = uri
                                break
                        
                        if post_uri:
                            embeddings[post_uri] = embedding
                            
                    except json.JSONDecodeError:
                        continue
        
        print(f"✅ Loaded {len(embeddings):,} embeddings")
        
        # Create embedded.jsonl - semantic content + embeddings
        embedded_file = self.datasets_dir / f"{run_name}_embedded.jsonl"
        embedded_count = 0

        print(f"📝 Creating {embedded_file}...")
        with open(embedded_file, 'w', encoding='utf-8') as f:
            for uri, embedding in embeddings.items():
                if uri in global_mapping:
                    semantic_content = global_mapping[uri]['semantic_content']

                    embedded_entry = {
                        'uri': uri,
                        'semantic_content': semantic_content,
                        'embedding': embedding
                    }

                    f.write(json.dumps(embedded_entry, ensure_ascii=False) + '\n')
                    embedded_count += 1

        print(f"✅ Created output file:")
        print(f"  📄 {embedded_file}: {embedded_count:,} entries (semantic content + embeddings)")

        return embedded_file
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists():
            print(f"🧹 Cleaning up temporary files...")
            shutil.rmtree(self.temp_dir)
            print(f"✅ Cleaned up {self.temp_dir}")


def main():
    parser = argparse.ArgumentParser(description='Embed semantic content for a Bluesky dataset run')
    parser.add_argument('run_name', help='Name of the run to embed (e.g., "august", "test")')
    parser.add_argument('--chunk-size', '-c', type=int, default=1000, 
                       help='Posts per chunk (default: 1000)')
    parser.add_argument('--check-interval', '-i', type=int, default=60,
                       help='Status check interval in seconds (default: 60 = 1 minute)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Overwrite existing output files without prompting')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY must be set in .env file")
        return 1
    
    processor = EmbeddingProcessor(api_key, args.chunk_size)
    
    try:
        print(f"🚀 Starting embedding process for '{args.run_name}'")
        print(f"{'='*60}")
        
        # Step 1: Validate input
        input_file = processor.validate_input(args.run_name, args.force)
        
        # Step 2: Prepare batches
        batch_data = processor.prepare_batch_chunks(input_file, args.run_name)
        
        # Step 3: Upload and create batch jobs
        batch_jobs = processor.upload_and_create_batches(batch_data)
        
        # Step 4: Monitor until completion
        success = processor.monitor_batches(batch_jobs, args.check_interval)
        
        # Step 5: Download results
        downloaded_files = processor.download_results(batch_jobs)
        
        if not downloaded_files:
            print("❌ No results downloaded - cannot create output files")
            return 1
        
        # Step 6: Create output file
        embedded_file = processor.create_output_files(batch_data, downloaded_files)
        
        # Step 7: Cleanup
        processor.cleanup()
        
        print(f"\n🎉 Embedding completed successfully!")
        print(f"{'='*60}")
        print(f"📊 Results:")
        print(f"  📄 Embedded data: {embedded_file}")
        
        if not success:
            print(f"\n⚠️  Warning: Some chunks failed. Check the output files for completeness.")
            return 2
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n👋 Interrupted by user")
        processor.cleanup()
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        processor.cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())