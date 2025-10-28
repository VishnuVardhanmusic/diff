import argparse
import os
from bs4 import BeautifulSoup

def extract_main_content(mhtml_content):
    # Parse the MHTML as HTML since most browsers export readable HTML inside
    soup = BeautifulSoup(mhtml_content, 'html.parser')
    main_div = soup.find('div', {'role': 'main', 'id': 'mh-main-content'})
    if not main_div:
        return "# Error: No <div role='main' id='mh-main-content'> found."

    stop_headings = {"Related checkers", "External guidance", "See also"}
    markdown_lines = []

    start_scraping = False
    for element in main_div.descendants:
        if element.name == 'h1':
            markdown_lines.append(f"# {element.get_text(strip=True)}\n")
            start_scraping = True

        elif start_scraping and element.name == 'h2':
            text = element.get_text(strip=True)
            if text in stop_headings:
                break
            markdown_lines.append(f"\n## {text}\n")

        elif start_scraping and element.name == 'p':
            text = element.get_text(strip=True)
            if text:
                markdown_lines.append(f"{text}\n")
                
        elif start_scraping and element.name == 'ul':
            parent_div = element.find_parent('div')
            if parent_div:
                div_text = parent_div.get_text(separator="\n", strip=True)
                if parent_div.ul:
                    first_ul = parent_div.ul
                    before_ul_text = ""
                    for content in parent_div.contents:
                        if content == first_ul:
                            break
                        if isinstance(content, str):
                            before_ul_text += content.strip() + " "
                        elif hasattr(content, 'get_text'):
                            before_ul_text += content.get_text(strip=True) + " "
                    if before_ul_text.strip():
                        markdown_lines.append(before_ul_text.strip() + "\n")

            items = element.find_all('li')
            for li in items:
                text = li.get_text(strip=True)
                if text:
                    markdown_lines.append(f"- {text}")
            markdown_lines.append("")  

        elif start_scraping and element.name == 'pre':
            code = element.get_text()
            markdown_lines.append(f"\n```c\n{code}\n```\n")

        elif start_scraping and element.name == 'div' and element.get('class') == ['home-footer']:
            break

    return '\n'.join(markdown_lines).strip()

def main():
    parser = argparse.ArgumentParser(description="Extract main content from .mhtml and save as .md")
    parser.add_argument("input_file", help="Path to the input .mhtml file")
    args = parser.parse_args()

    input_file = args.input_file
    if not input_file.endswith(".mhtml"):
        raise ValueError("Input file must have .mhtml extension")

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        mhtml_content = f.read()

    markdown_content = extract_main_content(mhtml_content)
    output_file = os.path.splitext(input_file)[0] + ".md"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"✅ Extracted content saved to: {output_file}")

if __name__ == "__main__":
    main()
