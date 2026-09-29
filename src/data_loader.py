from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
        PROJECT_ROOT
        / "data"
        / "Kapston_Home_Services_Company_Information_and_Customer_Documentation.pdf"
)


def load_pdf(path: Path = PDF_PATH) -> list[dict]:
    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(
            reader.pages,
            start=1
    ):
        text = page.extract_text()

        if text and text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages