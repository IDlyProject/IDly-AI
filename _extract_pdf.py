from pypdf import PdfReader

path = r"c:\Users\hunji\OneDrive\바탕 화면\국가안보공모전_논문_제출최종본_v11.pdf"
reader = PdfReader(path)
text = []
for page in reader.pages:
    text.append(page.extract_text() or "")
full = "\n".join(text)
with open("_pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(full)
print(len(reader.pages), "pages,", len(full), "chars")
