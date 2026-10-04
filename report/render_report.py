"""Render the two-page report from its reviewable Markdown source."""
from pathlib import Path
import re
from html import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent


def inline(text):
    text = escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="#244f6a">\1</link>', text)
    return text


def render():
    styles = {
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9.4, leading=12.5, spaceAfter=6),
        'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=18, leading=22, spaceAfter=9),
        'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=11, leading=14, spaceBefore=6, spaceAfter=5),
        'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.6, leading=11, alignment=TA_LEFT),
    }
    story, pending, rows = [], [], []
    def flush():
        if pending:
            story.append(Paragraph(inline(' '.join(pending)), styles['body']))
            pending.clear()
        if rows:
            clean = [r for r in rows if not all(re.fullmatch(r':?-+:?', c.strip()) for c in r)]
            widths = [242, 281] if len(clean[0]) == 2 else [118, 197, 208]
            table = Table([[Paragraph(inline(c), styles['cell']) for c in row] for row in clean], colWidths=widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#e9eef1')),
                ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),4),
                ('BOTTOMPADDING',(0,0),(-1,-1),4),
                ('LINEBELOW',(0,0),(-1,0),0.7,colors.HexColor('#6b7a85')),
                ('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#d5dce0')),
            ]))
            story.extend([table, Spacer(1,7)])
            rows.clear()
    for line in (ROOT/'report.md').read_text(encoding='utf-8').splitlines():
        if line == '<!-- pagebreak -->':
            flush()
            story.append(PageBreak())
        elif line.startswith('# '):
            flush()
            story.append(Paragraph(inline(line[2:]), styles['title']))
        elif line.startswith('## '):
            flush()
            story.append(Paragraph(inline(line[3:]), styles['heading']))
        elif line.startswith('|'):
            if pending:
                flush()
            rows.append([c.strip() for c in line.strip('|').split('|')])
        elif not line.strip():
            flush()
        else:
            pending.append(line.strip())
    flush()
    def footer(canvas, doc):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#59646c'))
        canvas.drawString(36,22,'Soup DPO on a Colab T4 | Huzyefah Wasim | 5 October 2026')
        canvas.drawRightString(A4[0]-36,22,str(doc.page))
    output = ROOT/'report.pdf'
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=36,leftMargin=36,
                      topMargin=32,bottomMargin=36, title='Soup DPO T4 experiment',
                      author='Huzyefah Wasim').build(story,onFirstPage=footer,onLaterPages=footer)
    pages = len(PdfReader(output).pages)
    if pages != 2:
        raise ValueError(f'Report must be exactly two pages; rendered {pages}')
    print(f'{output}: {pages} pages')


if __name__ == '__main__':
    render()
