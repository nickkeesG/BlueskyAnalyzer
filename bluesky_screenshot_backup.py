#!/usr/bin/env python3
"""
Improved Bluesky Post Screenshot Tool
Usage: python bluesky_screenshot_improved.py <post_url>

Key improvements:
- Better post targeting to avoid capturing wrong posts in reply threads
- Filters out posts with role="link" to find the actual target post
- Fails loudly if no valid target post is found
"""

import sys
import argparse
import re
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BlueskyScreenshotImproved:
    def __init__(self):
        self.auth_file = "bluesky_auth.json"
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def parse_post_url(self, url: str) -> dict:
        """Extract username and post ID from Bluesky URL"""
        # Handle both formats:
        # https://bsky.app/profile/username/post/postid
        # at://did:plc:xxx/app.bsky.feed.post/postid

        if url.startswith("at://"):
            print("❌ AT URI format not supported yet. Please use web URL format:")
            print("   https://bsky.app/profile/username/post/postid")
            return None

        # Extract from web URL
        pattern = r'https://bsky\.app/profile/([^/]+)/post/([^/?#]+)'
        match = re.match(pattern, url)

        if not match:
            print("❌ Invalid Bluesky URL format")
            print("   Expected: https://bsky.app/profile/username/post/postid")
            return None

        return {
            'username': match.group(1),
            'post_id': match.group(2),
            'url': url
        }

    def start_browser(self):
        """Start Playwright browser"""
        print("🚀 Starting browser...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=False,  # Non-headless for better JS compatibility
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )

    def login_if_needed(self):
        """Login to Bluesky if we don't have saved auth"""
        auth_path = Path(self.auth_file)

        if auth_path.exists():
            print("📱 Using saved authentication...")
            self.context = self.browser.new_context(
                storage_state=self.auth_file,
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        else:
            print("🔐 Need to login to Bluesky...")
            self.context = self.browser.new_context(
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            self._perform_login()

    def _perform_login(self):
        """Perform login using environment variables"""
        self.page = self.context.new_page()

        # Get credentials from environment variables
        username = os.getenv('BLUESKY_HANDLE')
        password = os.getenv('BLUESKY_WEB_PASSWROD')  # Note: keeping the typo as it exists in .env

        if not username or not password:
            print("❌ BLUESKY_HANDLE and BLUESKY_WEB_PASSWROD environment variables required")
            print("💡 Add them to your .env file")
            return False

        print(f"🔐 Logging in as {username}...")

        try:
            # Navigate to login page
            print("🌐 Going to Bluesky login page...")
            self.page.goto("https://bsky.app/", wait_until="networkidle")

            # Debug: Take a screenshot to see what the login page looks like
            print("📸 Taking debug screenshot of login page...")
            self.page.screenshot(path="debug_login_page.png")

            # Look for login form - using the actual button structure you found
            print("🔍 Looking for Sign in button...")
            self.page.wait_for_selector('button[aria-label="Sign in"]', timeout=10000)
            self.page.click('button[aria-label="Sign in"]')

            # Fill in credentials using the actual data-testid attributes
            print("📝 Filling in username...")
            self.page.wait_for_selector('[data-testid="loginUsernameInput"]')
            self.page.fill('[data-testid="loginUsernameInput"]', username)

            print("🔐 Filling in password...")
            self.page.fill('[data-testid="loginPasswordInput"]', password)

            # Look for login/next button (might be "Next" or "Sign in")
            print("🔍 Looking for login button...")
            login_buttons = [
                'button:has-text("Next")',
                'button:has-text("Sign in")',
                'button:has-text("Log in")',
                'button[type="submit"]'
            ]

            button_found = False
            for selector in login_buttons:
                try:
                    self.page.wait_for_selector(selector, timeout=3000)
                    self.page.click(selector)
                    print(f"✅ Clicked button: {selector}")
                    button_found = True
                    break
                except:
                    continue

            if not button_found:
                print("⚠️  No login button found, trying Enter key...")
                self.page.press('[data-testid="loginPasswordInput"]', 'Enter')

            # Wait for successful login (look for home timeline)
            print("⏳ Logging in...")

            # Debug: Take screenshot after clicking Next to see what happens
            self.page.wait_for_timeout(3000)  # Wait 3 seconds
            print("📸 Taking debug screenshot after login attempt...")
            self.page.screenshot(path="debug_after_login.png")

            # Check if we're back at the main Bluesky page (successful login)
            current_url = self.page.url
            if current_url == "https://bsky.app/" or "bsky.app" in current_url:
                print("✅ Login successful!")
            else:
                print(f"🔍 Current URL: {current_url}")
                print("💡 Login might have failed")
                return False

            # Save authentication state
            self.context.storage_state(path=self.auth_file)
            print(f"💾 Saved authentication to {self.auth_file}")

            return True

        except Exception as e:
            print(f"❌ Login failed: {e}")
            print("💡 Make sure your credentials are correct")
            return False

    def find_target_post_element(self, post_info: dict):
        """
        Find the correct post element, filtering out reply posts with role='link'
        """
        print(f"🎯 Looking for target post by @{post_info['username']} (ID: {post_info['post_id']})")

        # Try different selectors in order of specificity
        post_selectors = [
            f'[data-testid="postThreadItem-by-{post_info["username"]}"]',  # Most specific
            '[data-testid*="postThreadItem"]',  # Any postThreadItem
            '[data-testid="postThreadItem"]',   # Generic postThreadItem
            '[data-testid="feedItem"]',
            'article',
            '[role="article"]'
        ]

        for selector in post_selectors:
            try:
                print(f"🔍 Trying selector: {selector}")

                # Find all matching elements
                elements = self.page.query_selector_all(selector)

                if not elements:
                    print(f"❌ No elements found for selector: {selector}")
                    continue

                print(f"✅ Found {len(elements)} element(s) with selector: {selector}")

                # Filter out elements that have role="link" on the container div itself
                valid_elements = []

                for i, element in enumerate(elements):
                    role = element.get_attribute("role")
                    print(f"   Element {i+1}: role='{role}'")

                    if role != "link":
                        valid_elements.append(element)
                        print(f"   ✅ Element {i+1} is valid (no role='link')")
                    else:
                        print(f"   ❌ Element {i+1} rejected (has role='link')")

                if valid_elements:
                    if len(valid_elements) > 1:
                        print(f"📝 Found {len(valid_elements)} valid posts, taking the first one")
                    else:
                        print(f"🎯 Found exactly 1 valid post")

                    return valid_elements[0]
                else:
                    print(f"❌ All {len(elements)} elements had role='link' - continuing to next selector")

            except Exception as e:
                print(f"❌ Error with selector '{selector}': {e}")
                continue

        # If we get here, we couldn't find any valid post element
        print("🚨 CRITICAL ERROR: No valid target post found!")
        print("🚨 All found elements had role='link', indicating they are reply/thread items")
        print("🚨 The target post should NOT have role='link' on its container div")
        print("💡 Check debug_post_page.png to see the page structure")
        return None

    def screenshot_post(self, post_info: dict, output_path: str = None):
        """Navigate to post and take screenshot with improved targeting"""
        if not self.context:
            print("❌ Not authenticated")
            return False

        if not self.page:
            self.page = self.context.new_page()

        try:
            print(f"🌐 Navigating to post...")
            self.page.goto(post_info['url'], wait_until="networkidle")

            # Wait for post to load
            print("⏳ Waiting for post content to load...")
            self.page.wait_for_load_state('networkidle')

            # Wait longer for React to render
            print("⏳ Waiting for React components to render...")
            self.page.wait_for_timeout(5000)  # Wait 5 seconds for React

            # Wait for any text content to appear (indicating the post loaded)
            try:
                self.page.wait_for_function("document.body.innerText.length > 100", timeout=10000)
                print("✅ Content detected on page")
            except:
                print("⚠️ No significant content detected, but continuing...")

            # Debug: Take screenshot of the post page
            print("📸 Taking debug screenshot of post page...")
            self.page.screenshot(path="debug_post_page.png")

            # Use improved post element finding
            post_element = self.find_target_post_element(post_info)

            if not post_element:
                print("🚨 FAILED: Could not find valid target post element!")
                print("🚨 This usually means:")
                print("🚨 1. The post URL is incorrect")
                print("🚨 2. The post is part of a reply thread and all visible posts have role='link'")
                print("🚨 3. The page structure has changed")
                print("💡 Check debug_post_page.png to debug the issue")
                return False

            # Hide sticky navigation elements before screenshot
            print("🎨 Hiding sticky navigation elements...")
            self.page.add_style_tag(content="""
                div[style*='position: sticky'] {
                    display: none !important;
                }
                div[style*='position:sticky'] {
                    display: none !important;
                }
            """)

            # Wait a bit more for images to load and CSS to apply
            self.page.wait_for_timeout(2000)

            # Generate output filename if not provided
            if not output_path:
                output_path = f"bluesky_post_{post_info['username']}_{post_info['post_id']}_improved.png"

            # Take screenshot of just the post element
            print(f"📸 Taking screenshot of target post...")
            post_element.screenshot(path=output_path)

            print(f"✅ Screenshot saved to: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to screenshot post: {e}")
            return False

    def cleanup(self):
        """Clean up browser resources"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()


def main():
    parser = argparse.ArgumentParser(description='Take screenshot of a Bluesky post (improved version)')
    parser.add_argument('url', help='Bluesky post URL')
    parser.add_argument('-o', '--output', help='Output filename (optional)')

    args = parser.parse_args()

    # Create screenshot tool
    screenshot_tool = BlueskyScreenshotImproved()

    try:
        # Parse the URL
        post_info = screenshot_tool.parse_post_url(args.url)
        if not post_info:
            return 1

        print(f"🎯 Target: @{post_info['username']}'s post {post_info['post_id']}")

        # Start browser
        screenshot_tool.start_browser()

        # Login if needed
        screenshot_tool.login_if_needed()

        # Take screenshot
        success = screenshot_tool.screenshot_post(post_info, args.output)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        screenshot_tool.cleanup()


if __name__ == "__main__":
    sys.exit(main())