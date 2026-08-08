from playwright.sync_api import sync_playwright
from markdownify import markdownify as md
from bs4 import BeautifulSoup


url = "https://www.te.eg/web/guest/personal/services/mobile-call-services"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.goto(url, wait_until="networkidle")

    # Find the actual service blocks
    services = page.locator("div.journal-content-article")

    print("Number of services found:", services.count())

    all_markdown = "# خدمات الموبايل\n\n"

    for i in range(services.count()):

        # Get the HTML of one service
        html = services.nth(i).evaluate(
            "(element) => element.outerHTML"
        )

        # Parse the HTML
        soup = BeautifulSoup(html, "html.parser")

        # Get the title
        title = soup.select_one(".title")

        if not title:
            continue

        title = title.get_text(strip=True)

        # Get the description
        description = soup.select_one(".description")

        if description:
            description = description.get_text(" ", strip=True)
        else:
            description = ""

        # Add service to Markdown
        all_markdown += f"## {title}\n\n"

        if description:
            all_markdown += f"{description}\n\n"

    browser.close()


# Save clean Markdown
with open("mobile_services.md", "w", encoding="utf-8") as f:
    f.write(all_markdown)

print("Clean Markdown created!")