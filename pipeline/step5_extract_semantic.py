#!/usr/bin/env python3
"""
Step 5: Extract semantic content for embeddings
Creates a comprehensive text field combining all post content for embedding models.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class SemanticExtractor:
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Set up paths
        self.datasets_dir = Path('datasets')
        run_name = self.config['run_name']
        self.processed_file = self.datasets_dir / f"{run_name}_processed.jsonl"
    
    def extract_external_content(self, embed: Dict[str, Any]) -> Optional[str]:
        """Extract content from external links"""
        embed_type = embed.get('$type', '')
        
        if embed_type == 'app.bsky.embed.external#view':
            external = embed.get('external', {})
            if external:
                title = external.get('title', '').strip()
                description = external.get('description', '').strip()
                
                if title and description:
                    return f"Link: {title} - {description}"
                elif title:
                    return f"Link: {title}"
                elif description:
                    return f"Link: {description}"
        
        return None
    
    def extract_image_content(self, embed: Dict[str, Any]) -> Optional[str]:
        """Extract alt text from images"""
        embed_type = embed.get('$type', '')
        
        if embed_type == 'app.bsky.embed.images#view':
            images = embed.get('images', [])
            alt_texts = []
            
            for img in images:
                alt = img.get('alt', '').strip()
                if alt:
                    alt_texts.append(alt)
            
            if alt_texts:
                if len(alt_texts) == 1:
                    return f"Image: {alt_texts[0]}"
                else:
                    return f"Images: {' | '.join(alt_texts)}"
        
        return None
    
    def extract_video_content(self, embed: Dict[str, Any]) -> Optional[str]:
        """Extract alt text from videos"""
        embed_type = embed.get('$type', '')
        
        if embed_type == 'app.bsky.embed.video#view':
            alt = embed.get('alt', '').strip()
            if alt:
                return f"Video: {alt}"
            else:
                return "Video content"
        
        return None
    
    def extract_quote_content(self, embed: Dict[str, Any]) -> List[str]:
        """Extract content from quote posts"""
        embed_type = embed.get('$type', '')
        quote_parts = []
        
        if embed_type == 'app.bsky.embed.record#view':
            record = embed.get('record', {})
            if record:
                quoted_text = record.get('value', {}).get('text', '')
                author = record.get('author', {})
                author_name = author.get('displayName') or author.get('handle', '')
                
                if quoted_text:
                    if author_name:
                        quote_parts.append(f"Quoting @{author_name}: {quoted_text}")
                    else:
                        quote_parts.append(f"Quoting: {quoted_text}")
                
                # Extract embedded content from quoted post
                embeds = record.get('embeds', [])
                for quoted_embed in embeds:
                    for extract_fn in [self.extract_image_content, self.extract_video_content, self.extract_external_content]:
                        content = extract_fn(quoted_embed)
                        if content:
                            quote_parts.append(f"Quoted post {content.lower()}")
        
        elif embed_type == 'app.bsky.embed.recordWithMedia#view':
            # Handle quote posts with attached media
            record = embed.get('record', {})
            if record:
                nested_record = record.get('record', {})
                if nested_record:
                    quoted_text = nested_record.get('value', {}).get('text', '')
                    author = nested_record.get('author', {})
                    author_name = author.get('displayName') or author.get('handle', '')
                    
                    if quoted_text:
                        if author_name:
                            quote_parts.append(f"Quoting @{author_name}: {quoted_text}")
                        else:
                            quote_parts.append(f"Quoting: {quoted_text}")
        
        return quote_parts
    
    def extract_facets_content(self, post: Dict[str, Any]) -> List[str]:
        """Extract hashtags, mentions, and links from rich text facets"""
        facet_parts = []
        record = post.get('record', {})
        facets = record.get('facets', [])
        
        if not facets:
            return facet_parts
        
        hashtags = set()
        mentions = set()
        links = set()
        
        for facet in facets:
            features = facet.get('features', [])
            for feature in features:
                feature_type = feature.get('$type', '')
                
                if feature_type == 'app.bsky.richtext.facet#tag':
                    tag = feature.get('tag', '').strip()
                    if tag:
                        hashtags.add(f"#{tag}")
                
                elif feature_type == 'app.bsky.richtext.facet#mention':
                    did = feature.get('did', '').strip()
                    if did:
                        mentions.add(f"@{did}")
                
                elif feature_type == 'app.bsky.richtext.facet#link':
                    uri = feature.get('uri', '').strip()
                    if uri and not uri.startswith('https://bsky.app'):
                        links.add(uri)
        
        if hashtags:
            facet_parts.append(f"Hashtags: {' '.join(sorted(list(hashtags)))}")
        
        if mentions:
            facet_parts.append(f"Mentions: {' '.join(sorted(list(mentions)))}")
        
        if links:
            facet_parts.append(f"External links: {' '.join(sorted(list(links)))}")
        
        return facet_parts
    
    def extract_all_embed_content(self, embed: Dict[str, Any]) -> List[str]:
        """Extract content from any embed type"""
        content_parts = []
        embed_type = embed.get('$type', '')
        
        if embed_type == 'app.bsky.embed.recordWithMedia#view':
            # Handle quote post with media
            media = embed.get('media', {})
            if media:
                for extract_fn in [self.extract_image_content, self.extract_video_content]:
                    content = extract_fn(media)
                    if content:
                        content_parts.append(content)
            
            quote_content = self.extract_quote_content(embed)
            content_parts.extend(quote_content)
        
        else:
            # Handle individual embed types
            for extract_fn in [self.extract_quote_content, self.extract_external_content, self.extract_image_content, self.extract_video_content]:
                if extract_fn == self.extract_quote_content:
                    content = extract_fn(embed)
                    content_parts.extend(content)
                else:
                    content = extract_fn(embed)
                    if content:
                        content_parts.append(content)
        
        return content_parts
    
    def extract_semantic_content(self, post: Dict[str, Any]) -> str:
        """Extract all semantic content from a post for embedding"""
        content_parts = []
        
        # Main post text
        main_text = post.get('record', {}).get('text', '').strip()
        if main_text:
            content_parts.append(f"Post: {main_text}")
        
        # Parent context for replies
        parent_text = post.get('parentText', '').strip()
        reply_info = post.get('record', {}).get('reply', {})
        if parent_text:
            if reply_info:
                content_parts.append(f"Replying to: {parent_text}")
            else:
                content_parts.append(f"Context: {parent_text}")
        
        # Root context for threaded conversations
        root_text = post.get('rootText', '').strip()
        if root_text and root_text != parent_text:
            if reply_info:
                content_parts.append(f"Thread context: {root_text}")
            else:
                content_parts.append(f"Related context: {root_text}")
        
        # Author information
        author = post.get('author', {})
        author_name = author.get('displayName', '').strip()
        if not author_name:
            author_name = author.get('handle', '').strip()
        if author_name:
            content_parts.append(f"Author: {author_name}")
        
        # Rich text facets (hashtags, mentions, links)
        facet_content = self.extract_facets_content(post)
        content_parts.extend(facet_content)
        
        # Embed content (images, videos, quotes, links)
        embed = post.get('embed', {})
        if embed:
            embed_contents = self.extract_all_embed_content(embed)
            content_parts.extend(embed_contents)
        
        # Language context
        langs = post.get('record', {}).get('langs', [])
        if langs and langs[0] != 'en':
            content_parts.append(f"Language: {langs[0]}")
        
        return '\n'.join(content_parts)
    
    def run(self):
        """Extract semantic content and add to posts in-place"""
        print("Extracting semantic content for embeddings...")
        
        if not self.processed_file.exists():
            print("ERROR: Processed file not found. Run steps 1-4 first.")
            return False
        
        # Check if already completed
        sample_has_semantic = False
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Check first 5 posts
                    break
                line = line.strip()
                if line:
                    try:
                        post = json.loads(line)
                        if 'semantic_content' in post:
                            sample_has_semantic = True
                            break
                    except json.JSONDecodeError:
                        continue
        
        if sample_has_semantic:
            print("Semantic content already extracted, skipping...")
            return True
        
        temp_file = self.datasets_dir / f"{self.config['run_name']}_temp_step5.jsonl"
        
        total_posts = 0
        enriched_posts = 0
        
        try:
            with open(self.processed_file, 'r', encoding='utf-8') as infile:
                with open(temp_file, 'w', encoding='utf-8') as outfile:
                    for line_num, line in enumerate(infile, 1):
                        if line_num % 10000 == 0:
                            print(f"  Processed {line_num:,} posts...")
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            post = json.loads(line)
                            total_posts += 1
                            
                            # Extract semantic content
                            semantic_content = self.extract_semantic_content(post)
                            
                            if semantic_content.strip():
                                post['semantic_content'] = semantic_content
                                enriched_posts += 1
                            else:
                                post['semantic_content'] = ""
                            
                            outfile.write(json.dumps(post, ensure_ascii=False) + '\n')
                            
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Skipping malformed JSON on line {line_num}: {e}")
                            continue
            
            # Replace original with enriched version
            os.rename(temp_file, self.processed_file)
            
        except Exception as e:
            print(f"ERROR: Semantic extraction failed: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False
        
        # Results summary
        if total_posts > 0:
            percentage = (enriched_posts / total_posts) * 100
        else:
            percentage = 0
        
        print(f"✓ Semantic content extraction complete!")
        print(f"Total posts processed: {total_posts:,}")
        print(f"Posts with semantic content: {enriched_posts:,} ({percentage:.1f}%)")
        print(f"Updated file: {self.processed_file}")
        
        if self.processed_file.exists():
            file_size = self.processed_file.stat().st_size / (1024*1024)
            print(f"File size: {file_size:.1f} MB")
        
        return True


def main():
    extractor = SemanticExtractor()
    success = extractor.run()
    if not success:
        exit(1)


if __name__ == "__main__":
    main()