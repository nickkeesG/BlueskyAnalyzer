#!/usr/bin/env python3
"""
Bluesky Data Pipeline - Simplified Version
Main orchestrator script that runs all pipeline steps sequentially.
"""

import sys


class BlueskyPipeline:
    def run_step_1(self):
        """Step 1: Download raw data from Bluesky API"""
        print("\n--- Step 1: Download Data ---")
        from pipeline.step1_download_data import BlueskyDataDownloader
        downloader = BlueskyDataDownloader()
        downloader.run()
    
    def run_step_2(self):
        """Step 2: Filter posts with at least 10 likes"""
        print("\n--- Step 2: Filter by Likes ---")
        from pipeline.step2_filter_likes import LikesFilter
        filter_obj = LikesFilter()
        filter_obj.run()
    
    def run_step_3(self):
        """Step 3: Filter out spam and adult content"""
        print("\n--- Step 3: Filter Content ---")
        from pipeline.step3_filter_content import ContentFilter
        content_filter = ContentFilter()
        content_filter.run()
    
    def run_step_4(self):
        """Step 4: Enrich replies with parent/root context"""
        print("\n--- Step 4: Enrich Replies ---")
        from pipeline.step4_enrich_replies import ReplyEnricher
        enricher = ReplyEnricher()
        enricher.run()
    
    def run_step_5(self):
        """Step 5: Extract semantic content for embeddings"""
        print("\n--- Step 5: Extract Semantic Content ---")
        from pipeline.step5_extract_semantic import SemanticExtractor
        extractor = SemanticExtractor()
        extractor.run()
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("🚀 Starting Bluesky Data Pipeline")
        
        # Run all steps
        self.run_step_1()
        self.run_step_2() 
        self.run_step_3()
        self.run_step_4()
        self.run_step_5()
        
        print("\n✓ Pipeline completed successfully!")


def main():
    pipeline = BlueskyPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
