"""
Build publication-quality PDF for Deliverable 4: Short Paper
AI 630: Responsible AI | Lahore University of Management Sciences (LUMS)
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Running Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, "AI 630 RESPONSIBLE AI: RESEARCH PROJECT SHORT PAPER")
            self.drawRightString(612 - 54, 750, "LUMS MS AI — NOVEMBER 2026")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.6)
            self.line(54, 743, 612 - 54, 743)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.6)
        self.line(54, 45, 612 - 54, 45)
        
        self.setFont("Helvetica", 7.5)
        self.drawString(54, 32, "Group Project Deliverable 4 — Hammad, Ammar, Daud, Rizwan, Sarmad")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_str)
        self.restoreState()


def build_short_paper_pdf(output_pdf="Trust_Aware_Fraud_Detection_Short_Paper.pdf"):
    # 54pt margin = 0.75 in
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    PRIMARY = colors.HexColor("#0f172a")     # Slate 900
    SECONDARY = colors.HexColor("#1e40af")   # Blue 800
    ACCENT_BLUE = colors.HexColor("#2563eb") # Blue 600
    LIGHT_BG = colors.HexColor("#f8fafc")    # Slate 50
    CARD_BORDER = colors.HexColor("#cbd5e1") # Slate 300
    TEXT_DARK = colors.HexColor("#1e293b")   # Slate 800
    TEXT_MUTED = colors.HexColor("#64748b")  # Slate 500

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        alignment=1, # Center
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=ACCENT_BLUE,
        alignment=1,
        spaceAfter=4
    )

    author_style = ParagraphStyle(
        'DocAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_DARK,
        alignment=1,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=PRIMARY,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.3,
        leading=10.9,
        textColor=TEXT_DARK,
        spaceAfter=3
    )

    abstract_style = ParagraphStyle(
        'Abstract_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_DARK
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.3,
        leading=11.5,
        textColor=TEXT_DARK,
        leftIndent=12,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=TEXT_DARK,
        alignment=1
    )

    ref_style = ParagraphStyle(
        'RefStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.0,
        leading=9.0,
        textColor=TEXT_DARK,
        leftIndent=14,
        firstLineIndent=-14,
        spaceAfter=1.8
    )

    story = []
    note_style = ParagraphStyle('Note', parent=styles['Normal'], fontName='Helvetica', fontSize=7.0, leading=9.2, textColor=TEXT_MUTED, spaceBefore=3, spaceAfter=6)

    def make_table(rows, col_widths):
        data = [[Paragraph(f"<b>{h}</b>", table_header_style) for h in rows[0]]]
        for r in rows[1:]:
            data.append([Paragraph(str(c), table_cell_style) for c in r])
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), SECONDARY),
            ('GRID', (0,0), (-1,-1), 0.5, CARD_BORDER),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        return t

    # =========================================================================
    # HEADER
    # =========================================================================
    story.append(Paragraph("Trust-Aware Credit Card Fraud Detection via Calibrated Uncertainty and Contextual LLM Escalation", title_style))
    story.append(Paragraph("AI 630: Responsible AI — Project Progress Short Paper", subtitle_style))
    story.append(Paragraph("<b>Authors:</b> Hammad, Ammar, Daud, Rizwan, Sarmad<br/>Syed Babar Ali School of Science and Engineering (SBASSE), LUMS · November 2026", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=CARD_BORDER, spaceAfter=8))

    # =========================================================================
    # ABSTRACT
    # =========================================================================
    abstract_text = (
        "<b>Abstract—</b>Credit card fraud detection must score every transaction within a tight latency budget, under extreme class imbalance "
        "(~2% fraud), while staying reliable when transaction patterns shift. Fast tabular models meet the latency budget but are forced to output a "
        "decision even when they are uncertain; Large Language Models (LLMs) can reason over serialized transaction context but take seconds per call, "
        "which rules them out as a first-line filter. We present the design, implementation and preliminary evaluation of an <b>Uncertainty-Gated Cascade</b>. "
        "System-1 pairs a LightGBM risk scorer with a Monte Carlo (MC) Dropout network that estimates epistemic uncertainty. A dual gate (an aleatoric margin "
        "around the decision threshold plus an epistemic-variance limit) routes uncertain transactions to System-2, a contextual auditor that reads a "
        "serialized case brief. <b>At this progress stage System-2 is a rule-based proxy; LLM integration is pending.</b> On a synthetic 100,000-transaction "
        "benchmark with a strict temporal split (Jan–Aug train, Sep–Oct calibration, Nov–Dec test), the gate keeps <b>90.27%</b> of test traffic on System-1 "
        "(~17.3 ms single-transaction latency on CPU) and escalates <b>9.73%</b>, which would cut projected synchronous LLM latency by ~89% compared with "
        "sending every transaction to an LLM. LightGBM alone reaches 0.8426 ROC-AUC and 0.0056 ECE. The proxy System-2 raises fraud recall on the escalated "
        "set from 0.40 to 0.95 and removes 59 of 232 confident misses, but adds 995 false positives, lowering overall F1 from 0.234 to 0.172 (significant under "
        "bootstrap); the ROC-AUC difference is not significant. These results validate the routing mechanism and identify System-2 precision as the main "
        "target for the remaining work."
    )
    abstract_table = Table([[Paragraph(abstract_text, abstract_style)]], colWidths=[504])
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 0.8, CARD_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10), ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 6))

    # =========================================================================
    # 1. INTRODUCTION
    # =========================================================================
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Card networks expect authorization decisions within roughly 50–100 ms, and fraud is typically under 2% of volume. Standard tabular classifiers "
        "(gradient-boosted trees, neural networks) produce a score for every transaction, including those that look unlike anything in the training data. "
        "When patterns shift over time, such models can miss fraud while reporting high confidence—a <b>silent failure</b> that no downstream check catches.",
        body_style
    ))
    story.append(Paragraph(
        "Recent work serializes tabular records into text so that LLMs can reason over them [12–15]. LLMs can weigh context that a tree model cannot, but at "
        "1–3 s and a monetary cost per call they cannot score a full payment stream. We therefore propose a two-tier design: "
        "(1) <b>System-1</b>, a fast tabular scorer with an epistemic-uncertainty estimate; "
        "(2) a <b>Dual Gate</b> that automates a transaction only if its score is far from the decision threshold (aleatoric margin |p(x) − θ*|) "
        "<i>and</i> its epistemic variance σ_MC(x) is low; and "
        "(3) <b>System-2</b>, a contextual auditor applied only to the ~10% of transactions the gate escalates, which turns the transaction and its behavioral "
        "history into a readable case brief. Rather than dumping rejected cases on human analysts, System-2 resolves them with a stated rationale.",
        body_style
    ))

    # =========================================================================
    # 2. RELATED WORK
    # =========================================================================
    story.append(Paragraph("2. Related Work", h1_style))
    story.append(Paragraph(
        "<b>Tabular models and fraud.</b> Grinsztajn et al. [1] show that tree ensembles still outperform deep networks on typical tabular data. "
        "Gorishniy et al. [2] (FT-Transformer) and Hollmann et al. [3] (TabPFN) narrow this gap with specialised architectures, and Aminian et al. [4] "
        "propose FraudTransformer, a time-aware GPT-style model evaluated on HSBC transaction data. Jesus et al. [5] release the Bank Account Fraud suite and "
        "show that realistic fraud data is biased, imbalanced and dynamic over time. All of these models output a score for every input without saying "
        "when they should not be trusted; they inform our choice of a tree-based System-1, which we augment with an uncertainty gate.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Calibration, uncertainty and selective classification.</b> Guo et al. [6] show that modern networks are miscalibrated and propose temperature "
        "scaling. Gal and Ghahramani [7] interpret dropout at test time (MC Dropout) as approximate Bayesian inference, giving a cheap epistemic-uncertainty "
        "estimate. Geifman and El-Yaniv [8] formalise the risk–coverage trade-off of abstaining classifiers, and Pugnana et al. [9] benchmark selective deep "
        "networks. Angelopoulos and Bates [10] survey conformal prediction, and Chen et al. [11] apply prototype-conditioned conformal prediction (ProtoCP) to "
        "temporal transaction graphs. In this literature, abstained samples are simply handed off; we use the same tools for our gate but route abstentions to "
        "a contextual auditor instead.",
        body_style
    ))
    story.append(Paragraph(
        "<b>LLMs on structured data.</b> Hegselmann et al. [12] (TabLLM) show that serialized LLMs classify tabular data well in few-shot settings, and Li "
        "et al. [13] use LLMs for tabular anomaly detection. Tan et al. [14] (FinFRE-RAG) study LLM understanding of structured financial data for fraud "
        "detection with retrieval augmentation, and Lin et al. [15] use reinforcement learning to train LLMs for interpretable credit card fraud detection. "
        "These works apply the LLM to every case; we borrow their serialization and auditing ideas but apply them only to the escalated tail.",
        body_style
    ))

    # =========================================================================
    # 3. DATASET
    # =========================================================================
    story.append(Paragraph("3. Dataset &amp; Preprocessing", h1_style))
    story.append(Paragraph(
        "<b>Dataset.</b> We use the public Kaggle <i>Credit Card Fraud Detection 2026</i> dataset: 100,000 transactions time-stamped 1 Jan–31 Dec 2026 with "
        "35 interpretable features (amount, merchant category and country, channel, device, customer profile, rolling velocities such as "
        "<code>transactions_last_1h</code>, and spatial features such as <code>distance_from_home_km</code>). Fraud prevalence is 2.055% (2,055 cases: "
        "Card-Not-Present 1,054, Account-Takeover 447, Synthetic-Identity 309, Card-Present 245). <b>The data is synthetic:</b> the "
        "<code>transaction_day_of_week</code> field disagrees with the actual calendar date in every row, and each card maps one-to-one to a customer. "
        "Results should therefore be read as a controlled test of the pipeline, not as evidence of real-world performance.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Temporal split.</b> Records are sorted by timestamp and split by month: train Jan–Aug (66,388; 2.020% fraud), calibration/validation Sep–Oct "
        "(16,905; 2.141%) and test Nov–Dec (16,707; 2.107%). All encoders, scalers, thresholds and gate parameters are fitted on train or validation only. "
        "Monthly fraud rates are flat (1.87%–2.29%), so this is a temporal hold-out rather than a demonstrated distribution shift.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Preprocessing.</b> 41,709 missing <code>browser</code> values (in-store and ATM transactions) are imputed as <code>'None/In-Store'</code> instead of "
        "being dropped. Nine categorical features are ordinal-encoded (unseen test categories map to −1), 20 numerical features are standardised, and six "
        "binary flags are kept as-is. The training negative:positive ratio is 48.5; CreditNet is trained with pos_weight = 24.25 (half the ratio), "
        "and LightGBM is trained without class weighting.",
        body_style
    ))

    # =========================================================================
    # 4. METHODOLOGY
    # =========================================================================
    story.append(Paragraph("4. Methodology", h1_style))
    story.append(Paragraph(
        "<b>System-1.</b> A 300-tree LightGBM model produces the risk score p(x). Its decision threshold θ* = 0.08 maximises F1 on the validation set. "
        "LightGBM is deterministic and gives no epistemic-uncertainty estimate, so we pair it with CreditNet, an MLP [35→128→64→32→1] with dropout "
        "p = 0.25, used only for uncertainty: with dropout left active we run M = 20 stochastic passes [7] and take the standard deviation σ_MC(x). "
        "We also report CreditNet as a standalone classifier, with and without temperature scaling [6], as an ablation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Dual Gate.</b> A transaction is automated only if |p(x) − θ*| ≥ 0.03 <i>and</i> σ_MC(x) ≤ τ_σ = 0.0873 (the 94th percentile of validation "
        "σ_MC). Both values are chosen on the validation set to target ~90% automation; otherwise the transaction is escalated.",
        body_style
    ))
    story.append(Paragraph(
        "<b>System-2 (current proxy).</b> Each escalated transaction is serialized into an English case brief (transaction profile, velocity, distance "
        "anomalies, prior declines/chargebacks, and the System-1 score and gate trigger). In this progress version the brief is scored by a "
        "<b>hand-written rule-based proxy</b> that adds fixed weights for distance, velocity, amount ratio and risk flags. Its output is blended with the System-1 score "
        "as 0.65·p(x) + 0.35·p_rule(x). The rule weights and the blend weight were set by hand, not tuned on validation; replacing this proxy with an LLM is "
        "the main remaining task.",
        body_style
    ))

    # =========================================================================
    # 5. IMPLEMENTATION PROGRESS
    # =========================================================================
    story.append(Paragraph("5. Implementation Progress", h1_style))
    story.append(Paragraph(
        "Completed: (i) temporal data pipeline (<code>src/data_2026.py</code>); (ii) CreditNet with MC Dropout and L-BFGS temperature scaling "
        "(<code>src/models/</code>, <code>src/calibration/</code>, <code>src/uncertainty/</code>); (iii) case serializer and auditor "
        "(<code>src/system2/</code>). The auditor contains Gemini and OpenAI API connectors, but these have not yet been called or tested. "
        "(iv) Benchmark scripts (<code>experiments/run_audit_fixed_benchmark.py</code>, <code>experiments/verify_short_paper.py</code>); "
        "a separate rerun reproduced every reported number exactly.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Challenges.</b> (1) Standalone CreditNet was badly calibrated (ECE 0.23), which led us to use LightGBM for scoring and keep CreditNet only "
        "for uncertainty. (2) Our first gate (fixed confidence 0.85) escalated 43.8% of traffic; re-tuning the gate on validation brought this down to 9.73%. "
        "(3) Dropping missing <code>browser</code> rows would have removed most in-store traffic, so we imputed them instead.",
        body_style
    ))

    # =========================================================================
    # 6. PRELIMINARY RESULTS
    # =========================================================================
    story.append(Paragraph("6. Preliminary Results", h1_style))
    story.append(Paragraph("All results are on the Nov–Dec test set (16,707 transactions, 352 frauds), from a single training seed.", body_style))
    story.append(make_table([
        ["Configuration", "θ", "ROC-AUC", "PR-AUC", "F1", "ECE", "Silent FNs*", "Coverage"],
        ["1. CreditNet, uncalibrated", "0.50", "0.8381", "0.1298", "0.1591", "0.2298", "23", "100%"],
        ["2. CreditNet, temp. scaled (T = 0.77)", "0.50", "0.8381", "0.1298", "0.1591", "0.2006", "36", "100%"],
        ["3. LightGBM (System-1 scorer)", "0.08", "<b>0.8426</b>", "<b>0.1603</b>", "0.2344", "0.0056", "232", "100%"],
        ["4. Gate, automated subset only", "0.08", "0.8377", "0.1599", "<b>0.2533</b>", "<b>0.0047</b>", "168", "90.27%"],
        ["5. Full cascade (rule proxy)", "0.08", "0.8377", "0.1490", "0.1723", "0.0092", "173", "100%"],
    ], [140, 32, 53, 50, 42, 44, 75, 68]))
    story.append(Paragraph(
        "<i>Table 1. *Silent FN = fraud predicted legitimate with confidence ≥ 0.80. Counts are only comparable between rows with the same θ: at θ = 0.08 "
        "every miss has confidence ≥ 0.92, so for rows 3–5 the silent-FN count equals the total FN count. Row 4 is measured on a different (easier) subset "
        "than the other rows.</i>", note_style
    ))

    story.append(make_table([
        ["Escalated set (1,626 txns, 107 frauds)", "Fraud recall", "False positives", "Confident misses removed"],
        ["LightGBM score only", "0.402", "266", "—"],
        ["Cascade (rule proxy)", "0.953", "1,261", "59 (all now correctly flagged)"],
    ], [180, 90, 90, 144]))
    story.append(Paragraph(
        "<i>Table 2. Effect of System-2 on the escalated set. The proxy flags ~78% of escalated transactions as fraud: it catches 59 more frauds at the cost "
        "of 995 extra false alarms (~17 per extra fraud caught).</i>", note_style
    ))

    story.append(make_table([
        ["Latency (CPU)", "Batch-amortized", "Single transaction"],
        ["LightGBM", "0.063 ms/tx", "4.28 ms"],
        ["MC Dropout, 20 passes", "—", "12.97 ms"],
        ["System-1 total (LightGBM + MC Dropout)", "0.406 ms/tx", "≈ 17.3 ms"],
        ["Cascade with rule proxy", "0.576 ms/tx", "—"],
        ["Projected: cascade with live LLM (1.5 s/call, synchronous)", "—", "≈ 161.5 ms avg"],
        ["Projected: LLM on every transaction", "—", "1,500 ms"],
    ], [270, 117, 117]))
    story.append(Paragraph(
        "<i>Table 3. Batch-amortized = total time for the 16,707-row test set divided by its size. Projection: 0.9027 × 17.3 ms + 0.0973 × 1,500 ms "
        "≈ 161.5 ms (≈ 89% below LLM-on-everything). The 1.5 s per-call figure is an assumption, not a measurement.</i>", note_style
    ))

    chart_img = "experiments/results_2026/cascade_performance_comparison.png"
    if os.path.exists(chart_img):
        story.append(KeepTogether([
            Image(chart_img, width=504, height=120),
            Paragraph("<b>Figure 1:</b> Test-set comparison of the five configurations: (left) ROC-AUC and PR-AUC; (centre) ECE, lower is better; (right) F1.",
                      ParagraphStyle('FigCap', parent=styles['Normal'], fontName='Helvetica', fontSize=7.2, leading=9.5, textColor=TEXT_MUTED, alignment=1, spaceBefore=2, spaceAfter=4))
        ]))

    story.append(Paragraph(
        "<b>Statistical uncertainty.</b> Paired bootstrap (500 resamples), cascade minus LightGBM: ΔROC-AUC = −0.005 [95% CI −0.011, +0.002] (not significant); "
        "ΔPR-AUC = −0.011 [−0.022, +0.001] (not significant); <b>ΔF1 = −0.061 [−0.084, −0.040] (significantly worse)</b>. LightGBM's own ROC-AUC 95% CI is "
        "[0.827, 0.861].",
        body_style
    ))
    story.append(Paragraph(
        "<b>Findings.</b> "
        "(1) <i>Routing works as designed:</i> the gate, tuned only on validation, kept 90.27% of test traffic on System-1, close to the 90% target. "
        "Single-transaction System-1 latency (~17.3 ms) fits a 50–100 ms authorization budget. "
        "(2) <i>Gating improves the automated decisions:</i> on the automated subset, F1 rises from 0.2344 to 0.2533 and ECE falls from 0.0056 to 0.0047. "
        "(3) <i>The current System-2 trades precision for recall:</i> it removes 59 confident misses but adds 995 false positives, so the full cascade is "
        "significantly worse than LightGBM on F1. A useful System-2 must beat the LightGBM score on the escalated set, where LightGBM recall is only 0.40. "
        "(4) <i>Temperature scaling sharpened rather than softened CreditNet</i> (T = 0.77 &lt; 1). ECE improved (0.2298 → 0.2006), but sharpening pushes "
        "below-threshold scores toward 0, so more missed frauds cross the 0.80 confidence line (23 → 36). T &lt; 1 also appeared in an earlier run with "
        "pos_weight = 5 (T = 0.76), so we do not attribute it to class weighting; the cause is still open.",
        body_style
    ))

    # =========================================================================
    # 7. REMAINING WORK
    # =========================================================================
    story.append(Paragraph("7. Remaining Work", h1_style))
    story.append(Paragraph(
        "1. <b>Replace the rule proxy with an LLM</b> (e.g., Llama-3-8B-Instruct locally and a Gemini Flash API model) on the 1,626 escalated cases. "
        "Success criterion: higher precision than the proxy and higher recall than LightGBM's 0.40 on this set; also measure real per-call latency and cost.<br/>"
        "2. <b>Tune the System-2 blend weight and rules on validation</b> rather than by hand.<br/>"
        "3. <b>Calibration diagnosis:</b> Platt scaling and a pos_weight = 1 ablation to explain T &lt; 1; conformal risk control [10] for the gate thresholds.<br/>"
        "4. <b>Rigor:</b> multiple seeds with confidence intervals, matched-threshold or matched-recall comparisons, risk–coverage curves (AURC), and "
        "separate ablations of the aleatoric and epistemic gates.<br/>"
        "5. <b>External validity:</b> repeat the evaluation on a real-world benchmark (e.g., Bank Account Fraud [5]) with genuine temporal drift.",
        body_style
    ))

    # =========================================================================
    # 8. REFERENCES
    # =========================================================================
    story.append(Paragraph("8. References", h1_style))
    refs = [
        "[1] L. Grinsztajn, E. Oyallon, and G. Varoquaux, 'Why do tree-based models still outperform deep learning on tabular data?' in <i>NeurIPS Datasets and Benchmarks Track</i>, 2022.",
        "[2] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, 'Revisiting deep learning models for tabular data,' in <i>NeurIPS</i>, 2021.",
        "[3] N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter, 'TabPFN: A transformer that solves small tabular classification problems in a second,' in <i>ICLR</i>, 2023.",
        "[4] G. Aminian, A. Elliott, T. Li, et al., 'FraudTransformer: Time-aware GPT for transaction fraud detection,' arXiv:2509.23712, 2025.",
        "[5] S. Jesus, J. Pombal, D. Alves, A. F. Cruz, P. Saleiro, R. P. Ribeiro, J. Gama, and P. Bizarro, 'Turning the tables: Biased, imbalanced, dynamic tabular datasets for ML evaluation,' in <i>NeurIPS Datasets and Benchmarks Track</i>, 2022.",
        "[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On calibration of modern neural networks,' in <i>ICML</i>, 2017.",
        "[7] Y. Gal and Z. Ghahramani, 'Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,' in <i>ICML</i>, 2016.",
        "[8] Y. Geifman and R. El-Yaniv, 'Selective classification for deep neural networks,' in <i>NeurIPS</i>, 2017.",
        "[9] A. Pugnana, L. Perini, J. Davis, and S. Ruggieri, 'Deep neural network benchmarks for selective classification,' <i>Journal of Data-centric Machine Learning Research (DMLR)</i>, 2024.",
        "[10] A. N. Angelopoulos and S. Bates, 'A gentle introduction to conformal prediction and distribution-free uncertainty quantification,' arXiv:2107.07511, 2021.",
        "[11] X. Chen, S. Gong, et al., 'Temporal graph prototype-conditioned conformal prediction for fraud detection,' arXiv:2608.15768, 2026.",
        "[12] S. Hegselmann, A. Buendia, H. Lang, M. Agrawal, X. Jiang, and D. Sontag, 'TabLLM: Few-shot classification of tabular data with large language models,' in <i>AISTATS</i>, 2023.",
        "[13] A. Li, Y. Zhao, C. Qiu, M. Kloft, P. Smyth, M. Rudolph, and S. Mandt, 'Anomaly detection of tabular data using LLMs,' arXiv:2406.16308, 2024.",
        "[14] X. Tan, Y. Ma, et al., 'Understanding structured financial data with LLMs: A case study on fraud detection,' arXiv:2512.13040, 2025.",
        "[15] C. Lin, Y. Zhang, M. Ran, W. Xue, H. Fan, Y. Xu, Z. Wan, S. Han, Y. Guo, and J. Song, 'Reinforcement learning of large language models for interpretable credit card fraud detection,' arXiv:2601.05578, 2026.",
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Successfully compiled Short Paper PDF to: {output_pdf}")


if __name__ == "__main__":
    build_short_paper_pdf()
