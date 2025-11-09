from playwright.async_api import async_playwright
import base64
import sys
import asyncio
import time
from PIL import Image
import io


async def capture_and_extract_brand_node(website_url):
    # Capture screenshot
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        # Navigate to website with timeout
        await page.goto(website_url, timeout=3000000)

        time.sleep(5)

        # Take full page screenshot
        screenshot_bytes = await page.screenshot(full_page=True, type="png")
        await browser.close()

        # Convert to base64
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        # croping image to width 1920 since some websites render aditional white space
        img = Image.open(io.BytesIO(screenshot_bytes))

        width, height = img.size

        cropped_image = img.crop((0, 0, 1920, height))

        cropped_image.save("cropped_screenshot.png")


if __name__ == "__main__":
    website_url = sys.argv[1]
    asyncio.run(capture_and_extract_brand_node(website_url))
