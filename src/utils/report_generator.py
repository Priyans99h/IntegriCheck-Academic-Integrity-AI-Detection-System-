"""
IntegriCheck — Report Generator v6  (Turnitin-Style)
=====================================================
DROP-IN REPLACEMENT for: src/utils/report_generator.py

TWO REPORT TYPES (exactly like Turnitin):
  1. generate_plagiarism_report(data, output_path)
  2. generate_ai_report(data, output_path)

PLAGIARISM REPORT STRUCTURE (matches Turnitin PDF 1):
  Page 1  → Cover Page  (title, student, file info, pages/words/chars)
  Page 2  → Integrity Overview  (% similarity, match groups, top sources %, integrity flags)
  Page 3  → Top Sources  (numbered list with domain + % bar)
  Page 4+ → Document Submission View  (full text with colored highlights + numbered badges)

AI DETECTION REPORT STRUCTURE (matches Turnitin PDF 2):
  Page 1  → Cover Page
  Page 2  → AI Writing Overview  (% detected, detection groups, disclaimer, FAQ)
  Page 3+ → AI Writing Submission  (full text with cyan highlights)

HOW TO CALL (from flask_app/app.py or anywhere):
  from src.utils.report_generator import generate_plagiarism_report, generate_ai_report

  # Plagiarism
  plag_data = {
      'doc_title':      'My Essay',
      'student_name':   'Rohit Raskar',
      'submission_id':  'IC-2026-XXXX',
      'file_name':      'essay.docx',
      'file_size':      '19.4 KB',
      'page_count':     4,
      'word_count':     1062,
      'char_count':     6869,
      'similarity_pct': 4,
      'full_text':      '...the actual submitted text...',
      'match_groups': {
          'not_cited':         {'count': 3, 'pct': 3},
          'missing_quotation': {'count': 1, 'pct': 1},
          'missing_citation':  {'count': 0, 'pct': 0},
          'cited_and_quoted':  {'count': 0, 'pct': 0},
      },
      'database_pct': {
          'Internet':    4,
          'Publication': 1,
          'Student':     1,
      },
      'integrity_flags': 0,
      'top_sources': [
          {'rank': 1, 'type': 'Internet', 'domain': 'www.coursehero.com', 'pct': 2},
          {'rank': 2, 'type': 'Internet', 'domain': 'dspace.bracu.ac.bd', 'pct': 1},
          {'rank': 3, 'type': 'Internet', 'domain': 'research-information.bris.ac.uk', 'pct': 1},
      ],
      'highlights': [
          # Each highlight = a matched span in full_text
          {'start': 120, 'end': 210, 'source_idx': 0, 'category': 'not_cited', 'score': 0.82},
          {'start': 350, 'end': 420, 'source_idx': 1, 'category': 'missing_quotation', 'score': 0.71},
      ],
  }
  generate_plagiarism_report(plag_data, 'reports/Plagiarism_Report_XYZ.pdf')

  # AI Detection
  ai_data = {
      'doc_title':     'My Essay',
      'student_name':  'Rohit Raskar',
      'submission_id': 'IC-2026-XXXX',
      'file_name':     'essay.docx',
      'file_size':     '19.4 KB',
      'page_count':    4,
      'word_count':    1062,
      'char_count':    6869,
      'ai_pct':        55,
      'full_text':     '...the actual submitted text...',
      'ai_highlights': [
          {'start': 0,   'end': 800,  'type': 'ai_generated'},
          {'start': 900, 'end': 1400, 'type': 'ai_generated'},
      ],
    }
  generate_ai_report(ai_data, 'reports/AI_Writing_Report_XYZ.pdf')
"""

import os
from datetime import datetime
from reportlab.pdfgen import canvas

from reportlab.lib              import colors
from reportlab.lib.colors       import HexColor, white, black
from reportlab.lib.enums        import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes    import A4
from reportlab.lib.styles       import ParagraphStyle
from reportlab.lib.units        import mm
from reportlab.platypus         import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Flowable, KeepTogether,
)

# ── Page Layout ──────────────────────────────────────────────────────────────
PW, PH = A4          # 595.28 x 841.89 points
LM = RM = 18 * mm
TM = BM = 22 * mm
UW = PW - LM - RM   # usable width ~174mm

# ── Brand / UI Colors ────────────────────────────────────────────────────────
C = {
    'dark':       HexColor('#1a1a2e'),
    'blue':       HexColor('#0f3460'),
    'accent':     HexColor('#4fc3f7'),
    'red':        HexColor('#dc2626'),
    'orange':     HexColor('#f59e0b'),
    'green':      HexColor('#16a34a'),
    'grey':       HexColor('#6b7280'),
    'light_grey': HexColor('#9ca3af'),
    'text':       HexColor('#1f2937'),
    'bg':         HexColor('#f9fafb'),
    'border':     HexColor('#e5e7eb'),
    'cyan':       HexColor('#0891b2'),
    'cyan_bg':    HexColor('#cffafe'),
    'purple':     HexColor('#7c3aed'),
    'purple_bg':  HexColor('#ede9fe'),
    # Turnitin-like badge colors
    'badge_red':    HexColor('#dc2626'),
    'badge_orange': HexColor('#d97706'),
    'badge_blue':   HexColor('#2563eb'),
    'badge_green':  HexColor('#16a34a'),
    'badge_purple': HexColor('#7c3aed'),
    'turnitin_header': HexColor('#1e3a5f'),
}

# Match category config
CAT = {
    'not_cited':         {'color': HexColor('#dc2626'), 'label': 'Not Cited or Quoted',   'desc': 'Matches with neither in-text citation nor quotation marks'},
    'missing_quotation': {'color': HexColor('#d97706'), 'label': 'Missing Quotations',     'desc': 'Matches that are still very similar to source material'},
    'missing_citation':  {'color': HexColor('#2563eb'), 'label': 'Missing Citation',       'desc': 'Matches that have quotation marks, but no in-text citation'},
    'cited_and_quoted':  {'color': HexColor('#16a34a'), 'label': 'Cited and Quoted',       'desc': 'Matches with in-text citation present, but no quotation marks'},
}

# Source type colors & labels
SRCTYPE = {
    'Internet':    {'color': HexColor('#2563eb'), 'label': 'Internet'},
    'Publication': {'color': HexColor('#059669'), 'label': 'Publication'},
    'Student':     {'color': HexColor('#7c3aed'), 'label': 'Student papers'},
}

# Colors per source index (for numbered highlight badges)
SRC_COLORS = [
    HexColor('#dc2626'), HexColor('#2563eb'), HexColor('#d97706'),
    HexColor('#16a34a'), HexColor('#7c3aed'), HexColor('#0891b2'),
    HexColor('#db2777'), HexColor('#65a30d'), HexColor('#ea580c'),
    HexColor('#0284c7'),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _esc(t):
    return (t or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _score_color(pct):
    if pct >= 70: return C['red']
    if pct >= 40: return C['orange']
    if pct >= 15: return HexColor('#f59e0b')
    return C['green']


def _score_label(pct):
    if pct >= 70: return 'HIGH RISK'
    if pct >= 40: return 'MEDIUM RISK'
    if pct >= 15: return 'LOW RISK'
    return 'ACCEPTABLE'


def _ai_color(pct):
    if pct >= 50: return C['red']
    if pct >= 20: return C['orange']
    return C['green']


def _ai_label(pct):
    if pct >= 50: return 'HIGH AI RISK'
    if pct >= 20: return 'CAUTION'
    return 'LOW AI RISK'


def _now():
    return datetime.now().strftime('%b %d, %Y, %I:%M %p')


def _styles():
    return {
        'cover_title': ParagraphStyle('CoverTitle', fontSize=22, fontName='Helvetica-Bold',
                                      textColor=C['dark'], spaceAfter=3*mm, leading=26),
        'cover_sub':   ParagraphStyle('CoverSub',   fontSize=13, fontName='Helvetica-Bold',
                                      textColor=C['blue'], spaceAfter=2*mm, leading=17),
        'h2':          ParagraphStyle('H2', fontSize=16, fontName='Helvetica-Bold',
                                      textColor=C['dark'], spaceAfter=3*mm, leading=20),
        'h3':          ParagraphStyle('H3', fontSize=11, fontName='Helvetica-Bold',
                                      textColor=C['dark'], spaceAfter=2*mm, leading=14),
        'body':        ParagraphStyle('Body', fontSize=9, fontName='Helvetica',
                                      textColor=C['text'], leading=13, spaceAfter=1*mm),
        'body_j':      ParagraphStyle('BodyJ', fontSize=9, fontName='Helvetica',
                                      textColor=C['text'], leading=13, alignment=TA_JUSTIFY),
        'small':       ParagraphStyle('Small', fontSize=7.5, fontName='Helvetica',
                                      textColor=C['grey'], leading=10),
        'small_bold':  ParagraphStyle('SmallB', fontSize=7.5, fontName='Helvetica-Bold',
                                      textColor=C['grey'], leading=10),
        'label':       ParagraphStyle('Lbl', fontSize=8, fontName='Helvetica-Bold',
                                      textColor=C['grey'], leading=10),
        'value':       ParagraphStyle('Val', fontSize=8.5, fontName='Helvetica',
                                      textColor=C['text'], leading=12),
        'doc':         ParagraphStyle('Doc', fontSize=9.5, fontName='Helvetica',
                                      textColor=C['text'], leading=15, spaceAfter=3),
        'faq_q':       ParagraphStyle('FAQQ', fontSize=9, fontName='Helvetica-Bold',
                                      textColor=C['dark'], leading=12, spaceAfter=1*mm),
        'faq_a':       ParagraphStyle('FAQA', fontSize=8.5, fontName='Helvetica',
                                      textColor=C['text'], leading=12, spaceAfter=3*mm,
                                      alignment=TA_JUSTIFY),
        'disclaimer':  ParagraphStyle('Disc', fontSize=7.5, fontName='Helvetica',
                                      textColor=C['grey'], leading=10, alignment=TA_JUSTIFY),
        'center':      ParagraphStyle('Ctr', fontSize=9, fontName='Helvetica',
                                      textColor=C['text'], leading=13, alignment=TA_CENTER),
    }

# ── Numbered Canvas (for Page X of Y) ────────────────────────────────────────

class NumberedCanvas(canvas.Canvas):
    """
    Canvas that knows total page count.
    Allows: Page X of Y
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)

        for state in self._saved_page_states:
            self.__dict__.update(state)

            # Inject total pages into metadata
            if hasattr(self, '_page_deco'):
                self._page_deco.meta['total_pages'] = total_pages

            super().showPage()

        super().save()

# ── Custom Flowables ─────────────────────────────────────────────────────────

class _PageDeco:
    """Header + footer on every page."""
    def __init__(self, meta, report_type='Plagiarism'):
        self.meta        = meta
        self.report_type = report_type  # 'Plagiarism' or 'AI Writing'

    def __call__(self, canvas, doc):
        canvas.saveState()
        pg = canvas.getPageNumber()
        total = self.meta.get('total_pages', '?')
        sid   = self.meta.get('submission_id', '—')
        title = self.meta.get('doc_title', 'Submission')

        # ── Header ──
        canvas.setFillColor(C['turnitin_header'])
        canvas.rect(0, PH - 13*mm, PW, 13*mm, fill=1, stroke=0)

        # Logo placeholder (white box)
        canvas.setFillColor(white)
        canvas.setStrokeColor(white)
        canvas.roundRect(LM, PH - 10*mm, 22*mm, 7*mm, 2, fill=1, stroke=0)
        canvas.setFillColor(C['turnitin_header'])
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawString(LM + 2*mm, PH - 7*mm, 'IntegriCheck')

        canvas.setFillColor(white)
        canvas.setFont('Helvetica', 7.5)
        page_label = f'Page {pg} of {total} - {self.report_type} {"Overview" if pg == 2 else "Submission" if pg > 2 else "Cover Page"}'
        if pg == 1:
            page_label = f'Page {pg} of {total} - Cover Page'
        canvas.drawString(LM + 28*mm, PH - 7*mm, page_label)
        canvas.drawRightString(PW - LM, PH - 7*mm, f'Submission ID  {sid}')

        # ── Footer ──
        canvas.setFillColor(C['bg'])
        canvas.rect(0, 0, PW, 11*mm, fill=1, stroke=0)
        canvas.setStrokeColor(C['border'])
        canvas.setLineWidth(0.5)
        canvas.line(0, 11*mm, PW, 11*mm)
        canvas.setFillColor(C['grey'])
        canvas.setFont('Helvetica', 7)
        short_title = title[:40] + ('…' if len(title) > 40 else '')
        canvas.drawString(LM, 4*mm, f'IntegriCheck  ·  {short_title}')
        canvas.drawCentredString(PW / 2, 4*mm, f'Page {pg}  ·  Submission ID  {sid}')
        canvas.drawRightString(PW - LM, 4*mm, f'Generated: {self.meta.get("generated", "")}')
        canvas.restoreState()


class _BigPercentFlowable(Flowable):
    """Large percentage text with colored circle — for cover page."""
    def __init__(self, pct, color, label_line1, label_line2='', r=22*mm):
        Flowable.__init__(self)
        self.pct    = pct
        self.color  = color
        self.l1     = label_line1
        self.l2     = label_line2
        self.r      = r
        self.width  = r * 2 + 4*mm
        self.height = r * 2 + 14

    def draw(self):
        c  = self.canv
        cx = self.r + 2*mm
        cy = self.r + 8

        # Shadow circle
        c.setFillColor(HexColor('#d1d5db'))
        c.circle(cx + 1.5, cy - 1.5, self.r, fill=1, stroke=0)
        # Main circle
        c.setFillColor(self.color)
        c.circle(cx, cy, self.r, fill=1, stroke=0)
        # Inner white donut
        c.setFillColor(white)
        c.circle(cx, cy, self.r * 0.72, fill=1, stroke=0)
        # Percentage text
        c.setFillColor(self.color)
        fs = 28 if self.pct >= 10 else 32
        c.setFont('Helvetica-Bold', fs)
        c.drawCentredString(cx, cy + 4, f'{self.pct:.0f}%')
        # Risk badge inside circle
        lbl = _score_label(self.pct) if self.l1 == 'similarity' else _ai_label(self.pct)
        lw  = c.stringWidth(lbl, 'Helvetica-Bold', 6) + 10
        lh  = 9
        c.setFillColor(self.color)
        c.roundRect(cx - lw/2, cy - 13, lw, lh, 3, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont('Helvetica-Bold', 6)
        c.drawCentredString(cx, cy - 10, lbl)
        # Label below
        c.setFillColor(C['grey'])
        c.setFont('Helvetica', 8)
        c.drawCentredString(cx, 3, self.l2 or self.l1)


class _MatchGroupRow(Flowable):
    """One row in the Match Groups / Detection Groups table (icon + text + count + desc)."""
    def __init__(self, color, label, count, pct, desc, w=None):
        Flowable.__init__(self)
        self.color = color
        self.label = label
        self.count = count
        self.pct   = pct
        self.desc  = desc
        self.width  = w or UW
        self.height = 14*mm

    def draw(self):
        c = self.canv

        # Colored circle badge (left)
        c.setFillColor(self.color)
        c.circle(5*mm, 7*mm, 4*mm, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont('Helvetica-Bold', 7)
        count_str = str(self.count)
        c.drawCentredString(5*mm, 5.8*mm, count_str)

        # Label + count + pct
        x = 12*mm
        c.setFillColor(C['text'])
        c.setFont('Helvetica-Bold', 9.5)
        c.drawString(x, 8.5*mm, self.label)
        pct_str = f'  {self.pct}%'
        lw = c.stringWidth(self.label, 'Helvetica-Bold', 9.5)
        c.setFont('Helvetica', 9.5)
        c.setFillColor(C['grey'])
        c.drawString(x + lw, 8.5*mm, pct_str)

        # Description line
        c.setFont('Helvetica', 7.5)
        c.setFillColor(C['grey'])
        c.drawString(x, 4.5*mm, self.desc)

        # Bottom divider
        c.setStrokeColor(C['border'])
        c.setLineWidth(0.4)
        c.line(0, 0.5*mm, self.width, 0.5*mm)


class _SourceRow(Flowable):
    """One numbered source row with type badge, domain, and % bar."""
    def __init__(self, rank, src_type, domain, pct, w=None):
        Flowable.__init__(self)
        self.rank     = rank
        self.src_type = src_type
        self.domain   = domain
        self.pct      = pct
        self.width    = w or UW
        self.height   = 14*mm

    def draw(self):
        c    = self.canv
        col  = SRC_COLORS[(self.rank - 1) % len(SRC_COLORS)]
        tcol = SRCTYPE.get(self.src_type, {}).get('color', C['blue'])

        # Rank circle
        c.setFillColor(col)
        c.circle(5*mm, 7*mm, 4.5*mm, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(5*mm, 5.5*mm, str(self.rank))

        # Type badge pill
        badge_label = SRCTYPE.get(self.src_type, {}).get('label', self.src_type)
        bw = c.stringWidth(badge_label, 'Helvetica-Bold', 7) + 8
        bx = 12*mm
        c.setFillColor(tcol)
        c.roundRect(bx, 8*mm, bw, 5*mm, 2, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont('Helvetica-Bold', 7)
        c.drawString(bx + 4, 9.3*mm, badge_label)

        # Domain
        c.setFillColor(C['text'])
        c.setFont('Helvetica-Bold', 9)
        c.drawString(bx, 4.5*mm, self.domain)

        # Percentage + bar (right side)
        pct_str = f'{self.pct}%' if self.pct >= 1 else '<1%'
        bar_w   = self.width * 0.40
        bar_x   = self.width - bar_w
        fill_w  = max(bar_w * self.pct / 100, 4) if self.pct >= 1 else 4

        c.setFillColor(HexColor('#e5e7eb'))
        c.roundRect(bar_x, 5.5*mm, bar_w, 3*mm, 1.5, fill=1, stroke=0)
        c.setFillColor(col)
        c.roundRect(bar_x, 5.5*mm, fill_w, 3*mm, 1.5, fill=1, stroke=0)

        c.setFillColor(C['text'])
        c.setFont('Helvetica-Bold', 9)
        c.drawRightString(bar_x - 3, 6*mm, pct_str)

        # Divider
        c.setStrokeColor(C['border'])
        c.setLineWidth(0.4)
        c.line(0, 0.3*mm, self.width, 0.3*mm)


# ── Text Highlight Builder ────────────────────────────────────────────────────

def _build_highlighted_para(text, highlights, style, mode='plagiarism'):
    """
    Returns a Paragraph with inline highlights.
    mode='plagiarism' → colored underline + [N] badge per source
    mode='ai'         → cyan background highlight
    """
    if not text:
        return Paragraph('', style)
    if not highlights:
        return Paragraph(_esc(text), style)

    hl = sorted([h for h in highlights if h.get('end', 0) > h.get('start', 0)],
                key=lambda x: x['start'])

    # Merge overlapping spans
    merged = []
    for h in hl:
        if merged and h['start'] < merged[-1]['end']:
            if h.get('score', 0) > merged[-1].get('score', 0):
                merged[-1] = dict(h)
        else:
            merged.append(dict(h))

    parts = []
    cursor = 0
    for h in merged:
        s = max(h['start'], 0)
        e = min(h['end'], len(text))
        if s > cursor:
            parts.append(_esc(text[cursor:s]))
        span = _esc(text[s:e])
        if mode == 'plagiarism':
            idx  = h.get('source_idx', 0)
            cat  = h.get('category', 'not_cited')
            col  = CAT.get(cat, CAT['not_cited'])['color']
            hexc = f'#{int(col.red*255):02X}{int(col.green*255):02X}{int(col.blue*255):02X}'
            badge_col = SRC_COLORS[idx % len(SRC_COLORS)]
            hexb = f'#{int(badge_col.red*255):02X}{int(badge_col.green*255):02X}{int(badge_col.blue*255):02X}'
            parts.append(
                f'<font backcolor="#fff5f5" color="{hexc}"><b>{span}</b></font>'
                f'<super><font color="{hexb}" size="6"> [{idx+1}]</font></super>'
            )
        else:  # ai
            parts.append(
                f'<font backcolor="#cffafe" color="#0e7490"><b>{span}</b></font>'
            )
        cursor = e
    if cursor < len(text):
        parts.append(_esc(text[cursor:]))

    return Paragraph(''.join(parts), style)


def _split_into_paragraphs(text):
    """Split text into paragraphs on double newlines."""
    if not text:
        return ['']
    paras = [p.strip() for p in text.replace('\r\n', '\n').split('\n\n') if p.strip()]
    return paras or [text.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# PLAGIARISM REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_plagiarism_report(data: dict, output_path: str) -> str:
    """
    Generate a Turnitin-style Plagiarism PDF report.

    Parameters
    ----------
    data : dict  — see module docstring for full schema
    output_path : str — where to save the PDF

    Returns
    -------
    output_path (str)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    meta = {
        'doc_title':     data.get('doc_title', 'Untitled Document'),
        'submission_id': data.get('submission_id', f'IC-{datetime.now().strftime("%Y%m%d%H%M%S")}'),
        'generated':     _now(),
        'total_pages':   '?',
    }

    S  = _styles()
    story = []

    # ── PAGE 1: COVER ─────────────────────────────────────────────────────────
    sim_pct = float(data.get('similarity_pct', 0))

    # Spacer to push content down (Turnitin style — large top gap)
    story.append(Spacer(1, 55*mm))

    # Student name + title
    story.append(Paragraph(_esc(data.get('student_name', 'Unknown Student')), S['cover_title']))
    story.append(Paragraph(_esc(data.get('doc_title', 'Untitled')), S['cover_sub']))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph('Projects', S['body']))
    story.append(HRFlowable(width=UW, thickness=0.5, color=C['border'], spaceAfter=4*mm))

    # Document Details two-column table
    sid      = meta['submission_id']
    sub_date = data.get('submission_date', _now())
    dl_date  = data.get('download_date',  _now())
    fname    = data.get('file_name', 'document.docx')
    fsize    = data.get('file_size', '—')
    pg_count = data.get('page_count', '—')
    wc       = data.get('word_count', '—')
    cc       = data.get('char_count', '—')

    left_rows = [
        ('Submission ID',   sid),
        ('Submission Date', sub_date),
        ('Download Date',   dl_date),
        ('File Name',       fname),
        ('File Size',       fsize),
    ]
    right_rows = [
        (f'{pg_count} Pages',),
        (f'{wc} Words',),
        (f'{cc} Characters',),
    ]

    def _detail_cell(label, val):
        return [Paragraph(label, S['label']), Spacer(1, 0.5*mm),
                Paragraph(_esc(str(val)), S['value']), Spacer(1, 2.5*mm)]

    left_content  = []
    for lbl, val in left_rows:
        left_content += _detail_cell(lbl, val)

    right_content = []
    for (val,) in right_rows:
        right_content += [
            Paragraph(val, ParagraphStyle('RV', fontSize=10, fontName='Helvetica-Bold',
                                          textColor=C['dark'], leading=13)),
            Spacer(1, 3*mm),
        ]

    cover_table = Table(
        [[left_content, right_content]],
        colWidths=[UW * 0.62, UW * 0.38],
        style=TableStyle([
            ('VALIGN',      (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (1,0), (1,0),  8),
            ('BACKGROUND',  (1,0), (1,0),  C['bg']),
            ('BOX',         (1,0), (1,0),  0.5, C['border']),
        ])
    )
    story.append(cover_table)
    story.append(PageBreak())

    # ── PAGE 2: INTEGRITY OVERVIEW ────────────────────────────────────────────

    # Big % heading
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(f'<font size="36"><b>{sim_pct:.0f}%</b></font> &nbsp;Overall Similarity',
                           ParagraphStyle('BigPct', fontSize=14, fontName='Helvetica',
                                          textColor=C['dark'], leading=44)))
    story.append(Paragraph(
        'The combined total of all matches, including overlapping sources, for each database.',
        S['small']))
    story.append(Spacer(1, 5*mm))

    match_groups = data.get('match_groups', {})
    database_pct = data.get('database_pct', {})
    integrity_flags = int(data.get('integrity_flags', 0))
    top_sources  = data.get('top_sources', [])

    # Two-column layout: Match Groups (left) | Top Sources % (right)
    mg_items = []
    for key, cfg in CAT.items():
        info  = match_groups.get(key, {})
        count = info.get('count', 0)
        pct   = info.get('pct', 0)
        mg_items.append(_MatchGroupRow(cfg['color'], cfg['label'], count, pct, cfg['desc'], w=UW*0.52))

    ts_items = []
    ts_items.append(Paragraph('<b>Top Sources</b>', S['h3']))
    ts_items.append(Spacer(1, 2*mm))
    for db_key, db_label in [('Internet','Internet sources'),('Publication','Publications'),('Student','Submitted works (Student Papers)')]:
        pct = database_pct.get(db_key, 0)
        row = Table(
            [[Paragraph(f'{pct}%', ParagraphStyle('DP', fontSize=11, fontName='Helvetica-Bold',
                                                   textColor=C['text'], leading=13)),
              Paragraph(db_label, S['body'])]],
            colWidths=[14*mm, UW*0.38],
            style=TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE')])
        )
        ts_items.append(row)
        ts_items.append(Spacer(1, 1.5*mm))

    overview_table = Table(
        [[mg_items, ts_items]],
        colWidths=[UW * 0.56, UW * 0.44],
        style=TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(1,0),(1,0),8)])
    )
    story.append(Paragraph('<b>Match Groups</b>', S['h3']))
    story.append(Spacer(1, 2*mm))
    story.append(overview_table)
    story.append(Spacer(1, 6*mm))

    # Integrity Flags section
    story.append(HRFlowable(width=UW, thickness=0.4, color=C['border'], spaceAfter=3*mm))
    story.append(Paragraph('<b>Integrity Flags</b>', S['h3']))

    if integrity_flags == 0:
        flag_text = '0 Integrity Flags for Review'
        flag_color = C['green']
    else:
        flag_text = f'{integrity_flags} Integrity Flag{"s" if integrity_flags != 1 else ""} for Review'
        flag_color = C['red']

    flag_row = Table(
        [[Paragraph(f'<font color="#{int(flag_color.red*255):02X}{int(flag_color.green*255):02X}{int(flag_color.blue*255):02X}"><b>{flag_text}</b></font>', S['body']),
          Paragraph(
              'Our system\'s algorithms look deeply at a document for any inconsistencies that '
              'would set it apart from a normal submission. If we notice something strange, we flag '
              'it for you to review.\n\nA Flag is not necessarily an indicator of a problem. '
              'However, we\'d recommend you focus your attention there for further review.',
              S['small'])]],
        colWidths=[UW*0.4, UW*0.6],
        style=TableStyle([
            ('VALIGN',     (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING',(1,0), (1,0),   8),
            ('BACKGROUND', (1,0), (1,0),   C['bg']),
            ('BOX',        (1,0), (1,0),   0.4, C['border']),
            ('TOPPADDING', (1,0), (1,0),   6),
            ('BOTTOMPADDING',(1,0),(1,0),  6),
        ])
    )
    story.append(flag_row)
    story.append(PageBreak())

    # ── PAGE 3: TOP SOURCES ───────────────────────────────────────────────────
    story.append(Spacer(1, 4*mm))
    # Repeat match groups + database pct (like Turnitin page 3)
    story.append(Paragraph('<b>Match Groups</b>', S['h3']))
    story.append(Spacer(1, 1*mm))
    for key, cfg in CAT.items():
        info  = match_groups.get(key, {})
        count = info.get('count', 0)
        pct   = info.get('pct', 0)
        story.append(_MatchGroupRow(cfg['color'], cfg['label'], count, pct, cfg['desc']))

    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width=UW, thickness=0.4, color=C['border'], spaceAfter=3*mm))
    story.append(Paragraph('<b>Top Sources</b>', S['h3']))
    story.append(Paragraph('The sources with the highest number of matches within the submission. '
                           'Overlapping sources will not be displayed.', S['small']))
    story.append(Spacer(1, 3*mm))

    for src in top_sources:
        story.append(_SourceRow(src['rank'], src.get('type','Internet'),
                                src.get('domain','unknown.com'), src.get('pct',1)))
        story.append(Spacer(1, 1*mm))

    story.append(PageBreak())

    # ── PAGE 4+: DOCUMENT SUBMISSION VIEW ─────────────────────────────────────
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(_esc(data.get('doc_title', 'Document')), S['h2']))
    story.append(HRFlowable(width=UW, thickness=0.4, color=C['border'], spaceAfter=4*mm))

    full_text  = data.get('full_text', '')
    highlights = data.get('highlights', [])
    paragraphs = _split_into_paragraphs(full_text)

    cursor_pos = 0


    for para_text in paragraphs:
        offset = full_text.find(para_text, cursor_pos)

        if offset >= 0:
            cursor_pos = offset + len(para_text)

        if offset >= 0:
            cursor_pos = offset + len(para_text)
        if offset < 0:
            story.append(Paragraph(_esc(para_text), S['doc']))
            continue
        end_offset = offset + len(para_text)
        local_hl = []
        for h in highlights:
            hs = h.get('start', 0)
            he = h.get('end', 0)
            if hs < end_offset and he > offset:
                local_hl.append({
                    **h,
                    'start': max(hs - offset, 0),
                    'end':   min(he - offset, len(para_text)),
                })
        story.append(_build_highlighted_para(para_text, local_hl, S['doc'], mode='plagiarism'))

    # Source Legend
    if top_sources:
        story.append(Spacer(1, 6*mm))
        story.append(HRFlowable(width=UW, thickness=0.4, color=C['border'], spaceAfter=3*mm))
        story.append(Paragraph('<b>Source Legend</b>', S['h3']))
        legend_rows = []
        for src in top_sources:
            idx   = src['rank'] - 1
            col   = SRC_COLORS[idx % len(SRC_COLORS)]
            hexc  = f'#{int(col.red*255):02X}{int(col.green*255):02X}{int(col.blue*255):02X}'
            legend_rows.append(
                Paragraph(f'<font color="{hexc}"><b>[{src["rank"]}]</b></font>  '
                          f'{_esc(src.get("domain",""))}  —  {src.get("pct",0)}%', S['body'])
            )
        story.extend(legend_rows)

        # Category Legend
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph('<b>Match Category Legend</b>', S['h3']))
        for key, cfg in CAT.items():
            col  = cfg['color']
            hexc = f'#{int(col.red*255):02X}{int(col.green*255):02X}{int(col.blue*255):02X}'
            story.append(
                Paragraph(f'<font color="{hexc}">■</font>  '
                          f'<b>{cfg["label"]}</b>  —  {cfg["desc"]}', S['body'])
            )

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=LM, rightMargin=RM, topMargin=TM + 14*mm, bottomMargin=BM + 11*mm,
        title=meta['doc_title'],
    )
    deco = _PageDeco(meta, report_type='Integrity')
    def _decorate(canvas_obj, doc_obj):
        canvas_obj._page_deco = deco
        deco(canvas_obj, doc_obj)

    doc.build(
        story,
        onFirstPage=_decorate,
        onLaterPages=_decorate,
        canvasmaker=NumberedCanvas
    )
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# AI DETECTION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_ai_report(data: dict, output_path: str) -> str:
    """
    Generate a Turnitin-style AI Writing Detection PDF report.

    Parameters
    ----------
    data : dict  — see module docstring for full schema
    output_path : str — where to save the PDF

    Returns
    -------
    output_path (str)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    meta = {
        'doc_title':     data.get('doc_title', 'Untitled Document'),
        'submission_id': data.get('submission_id', f'IC-{datetime.now().strftime("%Y%m%d%H%M%S")}'),
        'generated':     _now(),
        'total_pages':   '?',
    }

    S     = _styles()
    story = []
    ai_pct = float(data.get('ai_pct', 0))

    # ── PAGE 1: COVER ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 55*mm))
    story.append(Paragraph(_esc(data.get('student_name', 'Unknown Student')), S['cover_title']))
    story.append(Paragraph(_esc(data.get('doc_title', 'Untitled')), S['cover_sub']))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph('Projects', S['body']))
    story.append(HRFlowable(width=UW, thickness=0.5, color=C['border'], spaceAfter=4*mm))

    sid      = meta['submission_id']
    sub_date = data.get('submission_date', _now())
    dl_date  = data.get('download_date',  _now())
    fname    = data.get('file_name', 'document.docx')
    fsize    = data.get('file_size', '—')
    pg_count = data.get('page_count', '—')
    wc       = data.get('word_count', '—')
    cc       = data.get('char_count', '—')

    left_rows = [
        ('Submission ID',   sid),
        ('Submission Date', sub_date),
        ('Download Date',   dl_date),
        ('File Name',       fname),
        ('File Size',       fsize),
    ]
    right_rows = [f'{pg_count} Pages', f'{wc} Words', f'{cc} Characters']

    def _dc(lbl, val):
        return [Paragraph(lbl, S['label']), Spacer(1, 0.5*mm),
                Paragraph(_esc(str(val)), S['value']), Spacer(1, 2.5*mm)]

    left_c = []
    for lbl, val in left_rows:
        left_c += _dc(lbl, val)
    right_c = []
    for val in right_rows:
        right_c += [Paragraph(val, ParagraphStyle('RV2', fontSize=10, fontName='Helvetica-Bold',
                                                   textColor=C['dark'], leading=13)),
                    Spacer(1, 3*mm)]

    story.append(Table(
        [[left_c, right_c]],
        colWidths=[UW*0.62, UW*0.38],
        style=TableStyle([
            ('VALIGN',      (0,0),(-1,-1),'TOP'),
            ('LEFTPADDING', (1,0),(1,0),  8),
            ('BACKGROUND',  (1,0),(1,0),  C['bg']),
            ('BOX',         (1,0),(1,0),  0.5, C['border']),
        ])
    ))
    story.append(PageBreak())

    # ── PAGE 2: AI WRITING OVERVIEW ───────────────────────────────────────────
    story.append(Spacer(1, 4*mm))

    # Big % + "detected as AI"
    ai_col = _ai_color(ai_pct)
    ai_hex = f'#{int(ai_col.red*255):02X}{int(ai_col.green*255):02X}{int(ai_col.blue*255):02X}'
    story.append(Paragraph(
        f'<font size="36" color="{ai_hex}"><b>{ai_pct:.0f}%</b></font>'
        f' <font size="22">detected as AI</font>',
        ParagraphStyle('AIPct', fontSize=14, fontName='Helvetica',
                       textColor=C['dark'], leading=44)
    ))
    story.append(Paragraph(
        'The percentage indicates the combined amount of likely AI-generated text as well as '
        'likely AI-generated text that was also likely AI-paraphrased.',
        S['small']))
    story.append(Spacer(1, 2*mm))

    # Caution banner (right-aligned box)
    caution_col = ai_col
    caution_hex = ai_hex
    caution_table = Table(
        [[Paragraph(
            f'<font color="{caution_hex}"><b>Caution: Review required.</b></font><br/>'
            '<font size="7.5" color="#6b7280">It is essential to understand the limitations of AI detection '
            'before making decisions about a student\'s work. We encourage you to learn more about '
            'IntegriCheck\'s AI detection capabilities before using the tool.</font>',
            S['small'])]],
        colWidths=[UW * 0.5],
        style=TableStyle([
            ('BACKGROUND',    (0,0),(0,0), C['bg']),
            ('BOX',           (0,0),(0,0), 0.5, C['border']),
            ('TOPPADDING',    (0,0),(0,0), 5),
            ('BOTTOMPADDING', (0,0),(0,0), 5),
            ('LEFTPADDING',   (0,0),(0,0), 6),
        ])
    )
    story.append(Table(
        [['', caution_table]],
        colWidths=[UW*0.45, UW*0.55],
        style=TableStyle([('VALIGN',(0,0),(-1,-1),'TOP')])
    ))
    story.append(Spacer(1, 5*mm))

    # Detection Groups
    story.append(Paragraph('<b>Detection Groups</b>', S['h3']))
    story.append(Spacer(1, 1*mm))

    ai_only_count = data.get('ai_only_count', 5)
    ai_para_count = 0
    story.append(_MatchGroupRow(C['cyan'], 'AI-generated only', ai_only_count, int(ai_pct),
                                'Likely AI-generated text from a large-language model.'))
    story.append(Spacer(1, 0.5*mm))
    story.append(_MatchGroupRow(C['purple'], 'AI-generated text that was AI-paraphrased',
                                ai_para_count, 0,
                                'Likely AI-generated text that was likely revised using an AI-paraphrase tool or word spinner.'))

    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width=UW, thickness=0.4, color=C['border'], spaceAfter=3*mm))

    # Disclaimer
    story.append(Paragraph('<b>Disclaimer</b>', S['small_bold']))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        'Our AI writing assessment is designed to help educators identify text that might be prepared '
        'by a generative AI tool. Our AI writing assessment may not always be accurate (i.e., our AI '
        'models may produce either false positive results or false negative results), so it should not '
        'be used as the sole basis for adverse actions against a student. It takes further scrutiny and '
        'human judgment in conjunction with an organization\'s application of its specific academic '
        'policies to determine whether any academic misconduct has occurred.',
        S['disclaimer']))

    story.append(Spacer(1, 4*mm))

    # FAQ
    faqs = [
        (
            'How should I interpret IntegriCheck\'s AI writing percentage and false positives?',
            'The percentage shown in the AI writing report is the amount of qualifying text within the '
            'submission that IntegriCheck\'s AI writing detection model determines was either likely '
            'AI-generated text from a large-language model or likely AI-generated text that was likely '
            'revised using an AI paraphrase tool or word spinner.\n\n'
            'False positives (incorrectly flagging human-written text as AI-generated) are a possibility '
            'in AI models.\n\n'
            'AI detection scores under 20%, which we do not surface in new reports, have a higher '
            'likelihood of false positives. To reduce the likelihood of misinterpretation, no score or '
            'highlights are attributed and are indicated with an asterisk in the report (*%).\n\n'
            'The AI writing percentage should not be the sole basis to determine whether misconduct has '
            'occurred. The reviewer/instructor should use the percentage as a means to start a formative '
            'conversation with their student and/or use it to examine the submitted assignment in '
            'accordance with their school\'s policies.'
        ),
        (
            'What does \'qualifying text\' mean?',
            'Our model only processes qualifying text in the form of long-form writing. Long-form writing '
            'means individual sentences contained in paragraphs that make up a longer piece of written '
            'work, such as an essay, a dissertation, or an article, etc. Qualifying text that has been '
            'determined to be likely AI-generated will be highlighted in cyan in the submission, and '
            'likely AI-generated and then likely AI-paraphrased will be highlighted purple.\n\n'
            'Non-qualifying text, such as bullet points, annotated bibliographies, etc., will not be '
            'processed and can create disparity between the submission highlights and the percentage shown.'
        ),
    ]
    story.append(Paragraph('<b>Frequently Asked Questions</b>', S['h3']))
    story.append(Spacer(1, 2*mm))
    for q, a in faqs:
        story.append(Paragraph(q, S['faq_q']))
        for line in a.split('\n\n'):
            story.append(Paragraph(line.strip(), S['faq_a']))

    story.append(PageBreak())

    # ── PAGE 3+: AI WRITING SUBMISSION ────────────────────────────────────────
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(_esc(data.get('doc_title', 'Document')), S['h2']))
    story.append(HRFlowable(width=UW, thickness=0.4, color=C['border'], spaceAfter=4*mm))

    full_text    = data.get('full_text', '')
    ai_highlights = data.get('ai_highlights', [])
    paragraphs   = _split_into_paragraphs(full_text)

    cursor_pos = 0

    for para_text in paragraphs:
        offset = full_text.find(para_text, cursor_pos)

        if offset >= 0:
            cursor_pos = offset + len(para_text)

        if offset < 0:
            story.append(Paragraph(_esc(para_text), S['doc']))
            continue
        end_offset = offset + len(para_text)
        local_hl = []
        for h in ai_highlights:
            hs = h.get('start', 0)
            he = h.get('end', 0)
            if hs < end_offset and he > offset:
                local_hl.append({
                    **h,
                    'start': max(hs - offset, 0),
                    'end':   min(he - offset, len(para_text)),
                })
        story.append(_build_highlighted_para(para_text, local_hl, S['doc'], mode='ai'))

    # AI Highlight Legend
    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width=UW, thickness=0.4, color=C['border'], spaceAfter=3*mm))
    story.append(Paragraph('<b>Highlight Legend</b>', S['h3']))
    story.append(Paragraph('<font color="#0e7490">■</font>  <u>Cyan underline</u>  — Likely AI-generated text', S['body']))
    story.append(Paragraph('<font color="#7c3aed">■</font>  <u>Purple underline</u>  — AI-generated + AI-paraphrased text', S['body']))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=LM, rightMargin=RM, topMargin=TM + 14*mm, bottomMargin=BM + 11*mm,
        title=meta['doc_title'],
    )
    deco = _PageDeco(meta, report_type='AI Writing')
    def _decorate(canvas_obj, doc_obj):
        canvas_obj._page_deco = deco
        deco(canvas_obj, doc_obj)

    doc.build(
        story,
        onFirstPage=_decorate,
        onLaterPages=_decorate,
        canvasmaker=NumberedCanvas
    )
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — old single function still works
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(data: dict, output_path: str) -> str:
    """
    Legacy wrapper — auto-detects report type.
    If data has 'ai_pct' key → AI report, else → Plagiarism report.
    """
    if 'ai_pct' in data:
        return generate_ai_report(data, output_path)
    return generate_plagiarism_report(data, output_path)




