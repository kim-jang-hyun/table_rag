"""
복잡한 테이블 유형 테스트용 PDF 생성 스크립트

생성되는 테이블 유형:
  E. 2단 열 헤더 (Multi-level Column Headers)
  F. 행/열 병합 혼재 (Rowspan + Colspan 혼재)
  G. 카테고리 행 + 소계 (Category Rows + Subtotals)
  H. 양방향 헤더 매트릭스 (Row + Column Headers)

실행:
  python create_complex_table_pdf.py
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
)

# ---------------------------------------------------------------------------
# 한국어 폰트 등록 (Windows Malgun Gothic)
# ---------------------------------------------------------------------------
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/malgunbd.ttf"
FONT_NAME = "Malgun"
FONT_BOLD_NAME = "MalgunBold"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
if os.path.exists(FONT_BOLD_PATH):
    pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, FONT_BOLD_PATH))

# ---------------------------------------------------------------------------
# 스타일
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

def _style(name, font=FONT_NAME, size=10, bold=False, align="LEFT", color=colors.black):
    return ParagraphStyle(
        name,
        fontName=FONT_BOLD_NAME if bold else FONT_NAME,
        fontSize=size,
        leading=size * 1.4,
        textColor=color,
        alignment={"LEFT": 0, "CENTER": 1, "RIGHT": 2}[align],
    )

title_style    = _style("TitleStyle",  size=14, bold=True,  align="CENTER")
section_style  = _style("SectionStyle", size=12, bold=True)
sub_style      = _style("SubStyle",    size=9,  align="CENTER")
body_style     = _style("BodyStyle",   size=9)
caption_style  = _style("CaptionStyle", size=8, color=colors.grey)

# ---------------------------------------------------------------------------
# 공통 테이블 스타일 헬퍼
# ---------------------------------------------------------------------------
HEADER_BG   = colors.HexColor("#2D5FA0")
SUBHDR_BG   = colors.HexColor("#5B8DD9")
CATEGORY_BG = colors.HexColor("#D9E4F5")
SUBTOTAL_BG = colors.HexColor("#BDD0EE")
TOTAL_BG    = colors.HexColor("#8FAADC")
ALT_BG      = colors.HexColor("#F2F6FC")
WHITE       = colors.white
BLACK       = colors.black

BASE_STYLE = [
    ("FONTNAME",    (0, 0), (-1, -1), FONT_NAME),
    ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
    ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#7F9DBF")),
    ("BOX",         (0, 0), (-1, -1), 1.2, colors.HexColor("#2D5FA0")),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",  (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0,0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",(0, 0), (-1, -1), 5),
]

def cell(text, bold=False, align="CENTER", size=8.5, color=colors.black, bg=None):
    """테이블 셀용 Paragraph 생성"""
    p = Paragraph(
        str(text),
        ParagraphStyle(
            "cell",
            fontName=FONT_BOLD_NAME if bold else FONT_NAME,
            fontSize=size,
            leading=size * 1.4,
            textColor=color,
            alignment={"LEFT": 0, "CENTER": 1, "RIGHT": 2}[align],
        ),
    )
    return p


def hdr(text, size=8.5):
    return cell(text, bold=True, align="CENTER", size=size, color=WHITE)


def num(text):
    return cell(str(text), align="RIGHT")


def lft(text, bold=False):
    return cell(text, bold=bold, align="LEFT")

# ---------------------------------------------------------------------------
# 섹션 E: 2단 열 헤더 테이블
# ---------------------------------------------------------------------------
def build_table_e():
    """
    부문별 실적 요약표
    헤더 Row 0: 부문(rowspan 2) | 매출(colspan 3) | 비용(colspan 3) | 이익(colspan 2)
    헤더 Row 1:                | 국내 | 해외 | 합계 | 재료비 | 인건비 | 합계 | 영업이익 | 순이익
    """
    data = [
        # Row 0 — 상위 헤더
        [hdr("부문"), hdr("매출 (백만원)"), "",          "",         hdr("비용 (백만원)"), "",        "",        hdr("이익 (백만원)"), ""],
        # Row 1 — 하위 헤더
        ["",          hdr("국내"),          hdr("해외"), hdr("합계"), hdr("재료비"),       hdr("인건비"), hdr("합계"), hdr("영업이익"),   hdr("순이익")],
        # 데이터 행
        [lft("반도체"),  num("45,000"), num("132,000"), num("177,000"), num("63,000"), num("28,500"), num("91,500"),  num("85,500"), num("71,200")],
        [lft("디스플레이"), num("28,000"), num("67,500"),  num("95,500"),  num("41,200"), num("19,800"), num("61,000"),  num("34,500"), num("28,100")],
        [lft("소재"),    num("31,500"), num("42,000"),  num("73,500"),  num("29,800"), num("16,200"), num("46,000"),  num("27,500"), num("22,400")],
        # 합계 행
        [cell("합계", bold=True, align="CENTER"), num("104,500"), num("241,500"), num("346,000"), num("134,000"), num("64,500"), num("198,500"), num("147,500"), num("121,700")],
    ]

    col_widths = [30*mm, 18*mm, 18*mm, 20*mm, 18*mm, 18*mm, 20*mm, 20*mm, 18*mm]

    style = TableStyle(BASE_STYLE + [
        # 상위 헤더 배경
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("BACKGROUND", (0, 1), (-1, 1), SUBHDR_BG),
        # 합계 행 배경
        ("BACKGROUND", (0, 5), (-1, 5), TOTAL_BG),
        ("FONTNAME",   (0, 5), (-1, 5), FONT_BOLD_NAME),
        ("TEXTCOLOR",  (0, 5), (-1, 5), WHITE),
        # 홀수 행 배경
        ("BACKGROUND", (0, 2), (-1, 2), ALT_BG),
        ("BACKGROUND", (0, 4), (-1, 4), ALT_BG),
        # 병합
        ("SPAN", (0, 0), (0, 1)),   # 부문 rowspan 2
        ("SPAN", (1, 0), (3, 0)),   # 매출 colspan 3
        ("SPAN", (4, 0), (6, 0)),   # 비용 colspan 3
        ("SPAN", (7, 0), (8, 0)),   # 이익 colspan 2
        # 굵은 구분선
        ("LINEAFTER",  (0, 0), (0, -1), 1.5, BLACK),
        ("LINEAFTER",  (3, 0), (3, -1), 1.5, BLACK),
        ("LINEAFTER",  (6, 0), (6, -1), 1.5, BLACK),
        ("LINEBELOW",  (0, 1), (-1, 1), 1.5, BLACK),
        ("LINEABOVE",  (0, 5), (-1, 5), 1.5, BLACK),
    ])

    return Table(data, colWidths=col_widths, style=style, repeatRows=2)


# ---------------------------------------------------------------------------
# 섹션 F: 행/열 병합 혼재 테이블
# ---------------------------------------------------------------------------
def build_table_f():
    """
    분기별 제품 실적표
    헤더 Row 0: 제품군(rowspan 2) | 제품(rowspan 2) | 1분기(colspan 3) | 2분기(colspan 3)
    헤더 Row 1:                   |                 | 실적 | 목표 | 달성률 | 실적 | 목표 | 달성률
    데이터: A계열 rowspan 2 (A-1, A-2), B계열 rowspan 2 (B-1, B-2)
    합계 행: 제품군+제품 colspan 2
    """
    data = [
        # Row 0
        [hdr("제품군"), hdr("제품"), hdr("1분기"), "", "", hdr("2분기"), "", ""],
        # Row 1
        ["", "", hdr("실적"), hdr("목표"), hdr("달성률"), hdr("실적"), hdr("목표"), hdr("달성률")],
        # A계열 - A-1
        [lft("A계열"), lft("A-1"), num("120"), num("100"), num("120.0%"), num("145"), num("150"), num("96.7%")],
        # A계열 - A-2
        ["",           lft("A-2"), num("85"),  num("90"),  num("94.4%"),  num("110"), num("100"), num("110.0%")],
        # B계열 - B-1
        [lft("B계열"), lft("B-1"), num("200"), num("210"), num("95.2%"),  num("195"), num("200"), num("97.5%")],
        # B계열 - B-2
        ["",           lft("B-2"), num("150"), num("140"), num("107.1%"), num("165"), num("160"), num("103.1%")],
        # 합계
        [cell("합계", bold=True, align="CENTER"), "", num("555"), num("540"), num("102.8%"), num("615"), num("610"), num("100.8%")],
    ]

    col_widths = [22*mm, 18*mm, 18*mm, 18*mm, 20*mm, 18*mm, 18*mm, 20*mm]

    style = TableStyle(BASE_STYLE + [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("BACKGROUND", (0, 1), (-1, 1), SUBHDR_BG),
        ("BACKGROUND", (0, 2), (-1, 2), ALT_BG),
        ("BACKGROUND", (0, 4), (-1, 4), ALT_BG),
        ("BACKGROUND", (0, 6), (-1, 6), TOTAL_BG),
        ("FONTNAME",   (0, 6), (-1, 6), FONT_BOLD_NAME),
        ("TEXTCOLOR",  (0, 6), (-1, 6), WHITE),
        # 병합
        ("SPAN", (0, 0), (0, 1)),   # 제품군 rowspan 2
        ("SPAN", (1, 0), (1, 1)),   # 제품 rowspan 2
        ("SPAN", (2, 0), (4, 0)),   # 1분기 colspan 3
        ("SPAN", (5, 0), (7, 0)),   # 2분기 colspan 3
        ("SPAN", (0, 2), (0, 3)),   # A계열 rowspan 2
        ("SPAN", (0, 4), (0, 5)),   # B계열 rowspan 2
        ("SPAN", (0, 6), (1, 6)),   # 합계 colspan 2
        # 구분선
        ("LINEAFTER",  (1, 0), (1, -1), 1.5, BLACK),
        ("LINEAFTER",  (4, 0), (4, -1), 1.5, BLACK),
        ("LINEBELOW",  (0, 1), (-1, 1), 1.5, BLACK),
        ("LINEBELOW",  (0, 3), (-1, 3), 1.2, colors.HexColor("#2D5FA0")),
        ("LINEABOVE",  (0, 6), (-1, 6), 1.5, BLACK),
    ])

    return Table(data, colWidths=col_widths, style=style, repeatRows=2)


# ---------------------------------------------------------------------------
# 섹션 G: 카테고리 행 + 소계 테이블
# ---------------------------------------------------------------------------
def build_table_g():
    """
    비용 상세 내역 — 직접비/간접비 카테고리 행 + 소계 + 총합
    """
    data = [
        # 헤더
        [hdr("비용 항목"), hdr("금액 (백만원)"), hdr("비중 (%)")],
        # 직접비 카테고리 (colspan 3)
        [cell("【직접비】", bold=True, align="LEFT", color=HEADER_BG), "", ""],
        [lft("  원재료비"),      num("28,500"), num("18.3%")],
        [lft("  노무비"),        num("14,200"), num("9.1%")],
        [lft("  외주비"),        num("8,700"),  num("5.6%")],
        [cell("  직접비 소계", bold=True, align="LEFT"), num("51,400"), num("33.0%")],
        # 간접비 카테고리
        [cell("【간접비】", bold=True, align="LEFT", color=HEADER_BG), "", ""],
        [lft("  감가상각비"),    num("32,100"), num("20.6%")],
        [lft("  판관비"),        num("41,500"), num("26.6%")],
        [lft("  R&D비용"),       num("17,800"), num("11.4%")],
        [lft("  기타"),          num("13,200"), num("8.5%")],
        [cell("  간접비 소계", bold=True, align="LEFT"), num("104,600"), num("67.0%")],
        # 총 합계
        [cell("총 합계", bold=True, align="CENTER"), num("156,000"), num("100.0%")],
    ]

    col_widths = [60*mm, 40*mm, 30*mm]

    style = TableStyle(BASE_STYLE + [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        # 카테고리 행
        ("BACKGROUND", (0, 1), (-1, 1), CATEGORY_BG),
        ("BACKGROUND", (0, 6), (-1, 6), CATEGORY_BG),
        # 소계 행
        ("BACKGROUND", (0, 5), (-1, 5), SUBTOTAL_BG),
        ("BACKGROUND", (0, 11), (-1, 11), SUBTOTAL_BG),
        ("FONTNAME",   (0, 5), (-1, 5), FONT_BOLD_NAME),
        ("FONTNAME",   (0, 11), (-1, 11), FONT_BOLD_NAME),
        # 총합 행
        ("BACKGROUND", (0, 12), (-1, 12), TOTAL_BG),
        ("FONTNAME",   (0, 12), (-1, 12), FONT_BOLD_NAME),
        ("TEXTCOLOR",  (0, 12), (-1, 12), WHITE),
        # 병합
        ("SPAN", (0, 1), (2, 1)),   # 직접비 카테고리 colspan 3
        ("SPAN", (0, 6), (2, 6)),   # 간접비 카테고리 colspan 3
        # 구분선
        ("LINEABOVE",  (0, 1), (-1, 1), 1.5, BLACK),
        ("LINEABOVE",  (0, 6), (-1, 6), 1.5, BLACK),
        ("LINEABOVE",  (0, 12), (-1, 12), 1.5, BLACK),
        ("LINEBELOW",  (0, 5), (-1, 5), 1.2, colors.HexColor("#2D5FA0")),
        ("LINEBELOW",  (0, 11), (-1, 11), 1.2, colors.HexColor("#2D5FA0")),
    ])

    return Table(data, colWidths=col_widths, style=style, repeatRows=1)


# ---------------------------------------------------------------------------
# 섹션 H: 양방향 헤더 매트릭스
# ---------------------------------------------------------------------------
def build_table_h():
    """
    지역별 제품 판매량 — 행 헤더(지역) + 열 헤더(제품) + 합계 행/열
    """
    data = [
        [hdr("지역"), hdr("A제품"), hdr("B제품"), hdr("C제품"), hdr("D제품"), hdr("합계")],
        [lft("서울"), num("1,250"), num("890"),   num("2,100"), num("450"),   num("4,690")],
        [lft("부산"), num("780"),   num("1,200"), num("890"),   num("320"),   num("3,190")],
        [lft("대구"), num("520"),   num("670"),   num("740"),   num("280"),   num("2,210")],
        [lft("광주"), num("430"),   num("580"),   num("620"),   num("190"),   num("1,820")],
        [cell("합계", bold=True, align="CENTER"),
         num("2,980"), num("3,340"), num("4,350"), num("1,240"), num("11,910")],
    ]

    col_widths = [22*mm, 25*mm, 25*mm, 25*mm, 25*mm, 25*mm]

    style = TableStyle(BASE_STYLE + [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("BACKGROUND", (0, 1), (0, -1), SUBHDR_BG),
        ("BACKGROUND", (0, 5), (-1, 5), TOTAL_BG),
        ("FONTNAME",   (0, 5), (-1, 5), FONT_BOLD_NAME),
        ("TEXTCOLOR",  (0, 5), (-1, 5), WHITE),
        # 홀수 데이터 행 배경
        ("BACKGROUND", (1, 2), (-1, 2), ALT_BG),
        ("BACKGROUND", (1, 4), (-1, 4), ALT_BG),
        # 구분선
        ("LINEAFTER",  (0, 0), (0, -1), 1.5, BLACK),
        ("LINEABOVE",  (0, 5), (-1, 5), 1.5, BLACK),
        ("LINEBELOW",  (0, 0), (-1, 0), 1.5, BLACK),
    ])

    return Table(data, colWidths=col_widths, style=style, repeatRows=1)


# ---------------------------------------------------------------------------
# PDF 조립
# ---------------------------------------------------------------------------
def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm,  bottomMargin=20*mm,
    )

    story = []

    # 문서 제목
    story.append(Paragraph("복잡한 테이블 유형 테스트 문서", title_style))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "본 문서는 RAG 파이프라인의 테이블 추출 정확도 검증을 위해 작성되었습니다.<br/>"
        "4가지 복잡한 테이블 구조(2단 열 헤더, 행/열 병합 혼재, 카테고리+소계, 양방향 매트릭스)를 포함합니다.",
        body_style,
    ))
    story.append(Spacer(1, 8*mm))

    # ── 섹션 E ──────────────────────────────────────────────────────────────
    story.append(Paragraph("E. 2단 열 헤더 테이블 (Multi-level Column Headers)", section_style))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "상위 헤더(매출/비용/이익)가 하위 헤더(국내/해외/합계 등)를 포함하는 2단 계층 구조입니다. "
        "'부문' 열은 2행에 걸쳐 병합(rowspan)되어 있습니다.",
        body_style,
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("표 E-1. 부문별 실적 요약표 (단위: 백만원)", caption_style))
    story.append(Spacer(1, 1*mm))
    story.append(build_table_e())
    story.append(Spacer(1, 10*mm))

    # ── 섹션 F ──────────────────────────────────────────────────────────────
    story.append(Paragraph("F. 행/열 병합 혼재 테이블 (Mixed Rowspan + Colspan)", section_style))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "제품군(A계열/B계열)이 2개 행에 걸쳐 병합(rowspan)되고, "
        "분기 헤더(1분기/2분기)가 3개 열에 걸쳐 병합(colspan)된 복합 구조입니다. "
        "합계 행의 첫 두 열도 병합(colspan)되어 있습니다.",
        body_style,
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("표 F-1. 분기별 제품 실적표 (단위: 개)", caption_style))
    story.append(Spacer(1, 1*mm))
    story.append(build_table_f())
    story.append(Spacer(1, 10*mm))

    # ── 섹션 G ──────────────────────────────────────────────────────────────
    story.append(Paragraph("G. 카테고리 행 + 소계 테이블 (Category Rows + Subtotals)", section_style))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "직접비/간접비 카테고리 행(colspan 3)이 그룹 구분선 역할을 하고, "
        "각 그룹 끝에 소계 행이 있으며, 최하단에 총 합계 행이 있는 구조입니다.",
        body_style,
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("표 G-1. 비용 상세 내역 (단위: 백만원)", caption_style))
    story.append(Spacer(1, 1*mm))
    story.append(build_table_g())
    story.append(Spacer(1, 10*mm))

    # ── 섹션 H ──────────────────────────────────────────────────────────────
    story.append(Paragraph("H. 양방향 헤더 매트릭스 (Bidirectional Header Matrix)", section_style))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "행 헤더(지역)와 열 헤더(제품)가 모두 존재하는 매트릭스 구조입니다. "
        "특정 값을 읽으려면 행·열 헤더를 동시에 참조해야 합니다. "
        "마지막 행과 열에 합계가 있습니다.",
        body_style,
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("표 H-1. 지역별 제품 판매량 (단위: 개)", caption_style))
    story.append(Spacer(1, 1*mm))
    story.append(build_table_h())
    story.append(Spacer(1, 6*mm))

    # 범례
    story.append(Paragraph(
        "※ 모든 수치는 테스트 목적의 가상 데이터입니다.",
        caption_style,
    ))

    doc.build(story)
    print(f"PDF 생성 완료: {output_path}")


if __name__ == "__main__":
    output = "복잡한_테이블_테스트_샘플.pdf"
    build_pdf(output)
