import argparse
import asyncio
import re
from pathlib import Path

import requests
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--outdir", default="/data/downloads")
    ap.add_argument("--profile", default="/data/browser_profile")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--login-wait", type=int, default=180)
    ap.add_argument("--feishu-file", action="store_true",
                    help="Enable Feishu file block detection for video/attachment downloads")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir=args.profile,
            headless=True,
            accept_downloads=True,
            downloads_path=str(outdir),
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )

        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        # --- trap for Feishu video/file stream URLs ---
        captured_stream_urls = []

        def on_response(response):
            url = response.url
            if ("download/video/" in url or "download/v2/" in url) and "cover" not in url:
                captured_stream_urls.append(url)

        page.on("response", on_response)

        print(f"Opening: {args.url}")
        await page.goto(
            args.url,
            wait_until="domcontentloaded",
            timeout=args.timeout * 1000,
        )

        shot = outdir / "page_or_login.png"
        await page.screenshot(path=str(shot), full_page=True)
        print(f"Screenshot saved: {shot}")
        print("If this is a login/QR page, copy or view the screenshot, scan it, then wait.")

        try:
            await page.wait_for_timeout(args.login_wait * 1000)
        except Exception:
            pass

        await page.screenshot(
            path=str(outdir / "after_login_wait.png"),
            full_page=True,
        )

        # --- Feishu file-block download ---
        if args.feishu_file:
            print("Feishu file mode: searching for file blocks ...")
            file_block = page.locator("[data-block-type='file']")
            if await file_block.count() > 0:
                video_btn = page.locator("[data-block-type='file'] .btn-preview").first
                if await video_btn.count() > 0:
                    print("Clicking preview button on file block ...")
                    await video_btn.click()
                    await page.wait_for_timeout(10_000)

            if captured_stream_urls:
                stream_url = captured_stream_urls[0]
                print(f"Captured stream URL: {stream_url}")

                cookies = await ctx.cookies()
                cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

                # Try to extract a filename from the page
                page_html = await page.content()
                name_match = re.search(r'class="file-name">([^<]+)</div>', page_html)
                filename = name_match.group(1).strip() if name_match else "download"
                target = outdir / filename

                print(f"Downloading: {filename}")
                headers = {
                    "Cookie": cookie_str,
                    "Referer": args.url,
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
                    ),
                }
                resp = requests.get(stream_url, headers=headers, stream=True, timeout=600)
                resp.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)

                file_size = target.stat().st_size
                print(f"Downloaded: {target} ({file_size / (1024 * 1024):.2f} MB)")
                await ctx.close()
                return

        # --- generic download selectors ---
        candidates = [
            "text=下载",
            "text=Download",
            "text=download",
            "a[download]",
            "button:has-text('下载')",
            "button:has-text('Download')",
        ]

        download = None

        for selector in candidates:
            try:
                loc = page.locator(selector).first

                if await loc.count() == 0:
                    continue

                print(f"Trying download selector: {selector}")

                async with page.expect_download(timeout=60_000) as dl_info:
                    await loc.click()

                download = await dl_info.value
                break

            except PlaywrightTimeoutError:
                print(f"No download triggered by: {selector}")
            except Exception as e:
                print(f"Selector failed {selector}: {e}")

        if download is None:
            if captured_stream_urls:
                stream_url = captured_stream_urls[0]
                print(f"Using captured stream URL as fallback: {stream_url}")
                cookies = await ctx.cookies()
                cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
                target = outdir / "download"
                headers = {
                    "Cookie": cookie_str,
                    "Referer": args.url,
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
                    ),
                }
                resp = requests.get(stream_url, headers=headers, stream=True, timeout=600)
                resp.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                file_size = target.stat().st_size
                print(f"Downloaded: {target} ({file_size / (1024 * 1024):.2f} MB)")
            else:
                print("No automatic download captured.")
                print("Current URL:", page.url)
                print("Use screenshots in /data/downloads to inspect page state.")
            await ctx.close()
            return

        suggested = download.suggested_filename or "download.bin"
        target = outdir / suggested
        await download.save_as(str(target))
        print(f"Downloaded: {target}")

        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
