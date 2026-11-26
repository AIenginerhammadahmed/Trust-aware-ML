import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
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
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, "TRUST-AWARE ML FOR FRAUD DETECTION: MASTER RESEARCH PROJECT BRIEF")
            self.drawRightString(612 - 54, 750, "LUMS MS AI — CORE A/A* TRACK")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.6)
            self.line(54, 743, 612 - 54, 743)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.6)
        self.line(54, 45, 612 - 54, 45)
        
        self.setFont("Helvetica", 7.5)
        self.drawString(54, 32, "Confidential — Academic Research Dossier | Hammad (Lead) · Ammar · Daud · Rizwan · Sarmad")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_str)
        self.restoreState()

def build_pdf(filename="Trust_Aware_ML_Project_Brief.pdf"):
    # 54pt margin = 0.75 in
    doc = SimpleDocTemplate(
        filename,
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
    TEXT_MUTED = colors.HexColor("#475569")  # Slate 600

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        spaceAfter=10
    )

    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.5,
        textColor=TEXT_MUTED
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=SECONDARY,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.4,
        leading=11.8,
        textColor=TEXT_DARK,
        spaceAfter=5
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.5,
        textColor=TEXT_DARK
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=TEXT_DARK
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10.2,
        textColor=PRIMARY
    )

    story = []

    # =========================================================
    # PAGE 1: COVER, EXECUTIVE SUMMARY & LITERATURE GAPS
    # =========================================================
    story.append(Paragraph("TRUST-AWARE MACHINE LEARNING FOR CREDIT CARD FRAUD DETECTION", title_style))
    story.append(Paragraph("An Uncertainty-Gated System-1 to System-2 Cascade for Safe, Calibrated, and Explainable Financial Decision-Making", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=SECONDARY, spaceAfter=6))

    meta_text = (
        "<b>Program:</b> LUMS MS AI — Advanced AI Research Project &nbsp;|&nbsp; "
        "<b>Target Venue:</b> CORE A / A* (NeurIPS, ICML, AAAI, ACM ICAIF)<br/>"
        "<b>Research Group:</b> Hammad (Lead / Presenter), Ammar, Daud, Muhammad Rizwan Kareem, Sarmad<br/>"
        "<b>Benchmark Suite:</b> Credit Card Fraud Detection (2024 Full-Year Dataset), IEEE-CIS Fraud Benchmark, European Cardholders"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Executive Summary & Research Motivation", h1_style))
    p1 = (
        "Financial institutions worldwide lose over <b>$32 Billion annually</b> to payment fraud. "
        "Production automated fraud screening systems operate under acute cost asymmetry: missing a single fraudulent transaction "
        "results in direct financial liability, chargeback penalties, and regulatory fines ($250+ average), whereas "
        "wrongfully declining a legitimate customer causes friction, reputational harm, and long-term customer churn ($15 average). "
        "Modern deep learning architectures routinely report near-perfect discriminative performance on tabular benchmarks, "
        "frequently exceeding <b>0.99 ROC-AUC</b>. However, this academic metric creates a dangerous <b>illusion of reliability</b> in live production."
    )
    story.append(Paragraph(p1, body_style))

    p2 = (
        "In production payment streams, transactions exhibit extreme class imbalance (~2% positive fraud rate or lower). "
        "Under such heavy skew, standard deep neural networks and Gradient Boosted Decision Trees (GBDTs) produce severely "
        "<b>uncalibrated probabilities</b>. Standard point-estimate models conflate <i>confidence with correctness</i>: "
        "when a model makes an incorrect prediction with >90% subjective confidence, it bypasses automated security filters entirely. "
        "We term these catastrophic failures <b>'Silent Killers'</b>."
    )
    story.append(Paragraph(p2, body_style))

    callout_data = [[
        Paragraph(
            "<b>💡 The Core Research Question:</b><br/>"
            "<i>\"How can we architect a trust-aware decision framework that pairs high-speed tabular screening with "
            "rigorous probability calibration, quantifies epistemic uncertainty to detect novel attack tactics, and selectively "
            "routes high-risk edge cases to a Retrieval-Augmented LLM Auditor without violating millisecond payment SLAs?\"</i>",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1.2, SECONDARY),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Critical Literature Gaps in Current SOTA (2022–2026)", h1_style))
    gaps_text = (
        "<b>Gap 1: Tabular Miscalibration Under Severe Class Skew (Pugnana &amp; Ruggieri, AAAI 2024; Guo et al., ICML 2017)</b><br/>"
        "Modern deep neural networks and tree ensembles are notoriously miscalibrated on tabular data. While calibration has been widely "
        "studied on balanced image and text benchmarks, existing techniques fail to calibrate the extreme minority fraud tail (&lt;2%). "
        "Consequently, predicted probabilities cannot be trusted by downstream automated decision engines.<br/><br/>"
        "<b>Gap 2: Latency and Hallucination Bottlenecks of Standalone LLMs (Tan et al., ACL 2026; Hegselmann et al., AISTATS 2023)</b><br/>"
        "Recent research demonstrates that Large Language Models (LLMs) can reason over serialized tabular transaction features. "
        "However, deploying standalone LLMs across 100% of incoming transaction streams is operationally impossible: LLM inference takes "
        "&gt;500ms (violating the &lt;10ms payment gateway SLA), generates unsustainable token costs, and hallucinates false positives under extreme skew.<br/><br/>"
        "<b>Gap 3: Missing Bridge from Selective Abstention to Automated Resolution (Geifman &amp; El-Yaniv, NeurIPS 2017)</b><br/>"
        "Selective classification frameworks provide mathematical trade-offs between coverage and risk. However, existing methods merely "
        "abstain and dump all rejected instances into manual human review queues. When rejection rates reach 10–15%, human investigators "
        "experience severe alert fatigue, neutralizing the operational benefit of automated machine learning."
    )
    story.append(Paragraph(gaps_text, body_style))

    # =========================================================
    # PAGE 2: MATHEMATICAL FOUNDATIONS & THEORETICAL RIGOR
    # =========================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Theoretical Foundations & Mathematical Formulations", h1_style))

    story.append(Paragraph("3.1 Probability Calibration & Expected Calibration Error (ECE)", h2_style))
    p_ece = (
        "A classification model is calibrated if its predicted probability reflects the true empirical likelihood of the event occurring: "
        "<b>P(Y = 1 | p_hat = p) = p</b> for all p in [0, 1]. In high-stakes fraud detection, we partition the probability range [0, 1] into M equal-width "
        "bins B_m. The <b>Expected Calibration Error (ECE)</b> is computed as the weighted average discrepancy between bin accuracy and confidence:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>ECE = &sum;_{m=1}^{M} (|B_m| / N) &middot; |acc(B_m) - conf(B_m)|</b><br/>"
        "Because 98% of transactions belong to the negative class, standard ECE is heavily diluted. We formulate <b>Class-Conditional ECE</b> "
        "(ECE_{y=1} for fraud and ECE_{y=0} for legitimate transactions) to independently measure reliability on the rare fraud tail."
    )
    story.append(Paragraph(p_ece, body_style))

    story.append(Paragraph("3.2 Post-Hoc Temperature Scaling (Guo et al., ICML 2017)", h2_style))
    p_temp = (
        "To calibrate logit outputs without altering model ranking or discriminative AUC, we optimize a single positive scalar temperature parameter T* &gt; 0 "
        "on the held-out validation split. Given raw logit z(x), the calibrated probability is:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>p_hat(x) = &sigma;(z(x) / T*) = 1 / (1 + exp(-z(x) / T*))</b><br/>"
        "The optimal scalar T* is learned via the L-BFGS optimizer by minimizing validation Negative Log-Likelihood (NLL) while keeping network weights W frozen:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>T* = argmin_{T &gt; 0} -&sum;_{i=1}^{N_val} [ y_i log &sigma;(z_i / T) + (1 - y_i) log(1 - &sigma;(z_i / T)) ]</b><br/>"
        "Because division by T* is a strictly monotonic transformation, ROC-AUC and PR-AUC are guaranteed to remain 100% preserved."
    )
    story.append(Paragraph(p_temp, body_style))

    story.append(Paragraph("3.3 Bayesian Epistemic Uncertainty via Monte Carlo Dropout (Gal &amp; Ghahramani, ICML 2016)", h2_style))
    p_mc = (
        "Standard neural networks produce deterministic point estimates, blinding them to whether an error is caused by data noise (aleatoric risk) "
        "or lack of familiarity with an emerging fraud pattern (epistemic ignorance). We keep Dropout (p=0.3) active at inference across T=50 stochastic "
        "forward passes to sample from the approximate weight posterior distribution:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Predictive Mean: p_bar(x) = (1 / T) &sum;_{t=1}^{T} &sigma;(f_{W_t}(x))</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Epistemic Standard Deviation: &sigma;_MC(x) = &radic;[ (1 / T) &sum;_{t=1}^{T} (p_t(x) - p_bar(x))&sup2; ]</b><br/>"
        "When a transaction exhibits a familiar pattern, network weights reach consensus and &sigma;_MC &approx; 0. When an attacker deploys a novel, zero-day "
        "synthetic identity or proxy tactic, weights disagree violently and &sigma;_MC spikes, signaling model ignorance."
    )
    story.append(Paragraph(p_mc, body_style))

    story.append(Paragraph("3.4 Cost-Sensitive Selective Risk & Rejection Dynamics (Geifman &amp; El-Yaniv, NeurIPS 2017)", h2_style))
    p_sel = (
        "Selective classification introduces a selection function g(x) in {0, 1}. Coverage c is the fraction of transactions automated by the model: "
        "c = (1/N) &sum; g(x_i). Selective risk R(c) is the error rate strictly on the accepted subset. We formulate the operational objective under a financial loss matrix:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Total Financial Loss = C_FN &middot; FN + C_FP &middot; FP + C_audit &middot; Audit</b><br/>"
        "where C_FN = $250 (direct fraud liability), C_FP = $15 (customer friction/churn), and C_audit = $5 (human review cost)."
    )
    story.append(Paragraph(p_sel, body_style))

    story.append(Paragraph("3.5 Finite-Sample Conformal Prediction Sets (Angelopoulos &amp; Bates, 2023)", h2_style))
    p_conf = (
        "To establish mathematical distribution-free coverage guarantees, we compute split conformal prediction sets C(x) &sube; {0, 1} "
        "such that P(Y in C(x)) &ge; 1 - &alpha; for a user-specified significance level &alpha; &isin; (0, 1). Transactions with ambiguous prediction sets "
        "C(x) = {0, 1} are identified as high-risk edge cases and routed directly into our secondary auditing pipeline."
    )
    story.append(Paragraph(p_conf, body_style))

    # =========================================================
    # PAGE 3: PROPOSED ARCHITECTURE & DATASET SUITE
    # =========================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Proposed Architecture: System-1 to System-2 Trust Cascade", h1_style))
    p_arch = (
        "Our core architectural contribution is an <b>Uncertainty-Gated Cognitive Cascade</b>. "
        "System-1 provides millisecond automated screening for bulk traffic, while System-2 provides deep semantic reasoning on the uncertain tail:"
    )
    story.append(Paragraph(p_arch, body_style))

    arch_table_data = [
        [
            Paragraph("<b>Component</b>", table_header_style),
            Paragraph("<b>Latency</b>", table_header_style),
            Paragraph("<b>Operational Role &amp; Methodology</b>", table_header_style)
        ],
        [
            Paragraph("<b>System-1 Fast Path</b><br/>(CreditNet / XGBoost)", table_cell_bold),
            Paragraph("&lt; 2 ms", table_cell_style),
            Paragraph("Screens 100% of streaming transactions. Outputs calibrated probability p_hat via Temperature Scaling (T*) and extracts epistemic uncertainty &sigma;_MC via 50 parallelized MC Dropout passes.", table_cell_style)
        ],
        [
            Paragraph("<b>Dual Trust Gatekeeper</b><br/>(Selective Policy)", table_cell_bold),
            Paragraph("&lt; 0.5 ms", table_cell_style),
            Paragraph("Evaluates two conditions: (1) Aleatoric check: Conf &ge; &tau;_conf; (2) Epistemic check: &sigma;_MC &le; &tau;_&sigma;. If BOTH pass (~88-90% volume), the decision is executed immediately without invoking LLM compute.", table_cell_style)
        ],
        [
            Paragraph("<b>System-2 Slow Path</b><br/>(Table-to-Text RAG-LLM)", table_cell_bold),
            Paragraph("~350–500 ms<br/>(Only on 10% tail)", table_cell_style),
            Paragraph("Serializes the ~10% deferred rows into natural language sentences. Retrieves k=4 balanced historical exemplars (2 fraud, 2 legit) from train memory via FinFRE-RAG (ACL 2026). Resolves ~80% of edge cases autonomously.", table_cell_style)
        ],
        [
            Paragraph("<b>Tier-3 Human Review</b><br/>(Compliance Investigators)", table_cell_bold),
            Paragraph("Human Queue<br/>(&lt; 2% volume)", table_cell_style),
            Paragraph("Fraud analysts receive only the residual ~2% ultra-complex edge cases, accompanied by the LLM-generated narrative audit explanation citing exact risk factors.", table_cell_style)
        ]
    ]

    arch_table = Table(arch_table_data, colWidths=[120, 74, 310])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.6, CARD_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5. Dataset Suite & Temporal Concept Drift Protocol", h1_style))
    p_data = (
        "Addressing the critical feedback regarding outdated, anonymized PCA datasets, our project upgrades to a "
        "<b>modern multi-benchmark suite</b> featuring rich semantic and behavioral attributes:"
    )
    story.append(Paragraph(p_data, body_style))

    data_table_data = [
        [
            Paragraph("<b>Dataset Benchmark</b>", table_header_style),
            Paragraph("<b>Scale &amp; Imbalance</b>", table_header_style),
            Paragraph("<b>Feature Taxonomy &amp; Scientific Role</b>", table_header_style)
        ],
        [
            Paragraph("<b>Primary Benchmark:</b><br/>Credit Card Fraud Detection (2024 Full-Year Dataset)", table_cell_bold),
            Paragraph("Full Year 2024<br/>~2.06% Fraud", table_cell_style),
            Paragraph("Contains interpretable spatial features (<i>distance_from_home_km</i>), velocity (<i>transactions_last_1h</i>), commercial domains (<i>merchant_category</i>), and amounts. Enables authentic chronological splits (Jan–Aug Train, Sep–Oct Val, Nov–Dec Test) to test holiday concept drift.", table_cell_style)
        ],
        [
            Paragraph("<b>Industry Benchmark:</b><br/>IEEE-CIS Fraud Detection Benchmark (Vesta Corp)", table_cell_bold),
            Paragraph("590,000+ Transactions<br/>~3.5% Fraud", table_cell_style),
            Paragraph("Gold-standard industry benchmark combining transaction data with device identity attributes (OS, browser, device type). Tests epistemic uncertainty detection when transactions originate from novel, unseen device/browser manifolds.", table_cell_style)
        ],
        [
            Paragraph("<b>Numerical Ablation:</b><br/>European Cardholders (2013 Classic Benchmark)", table_cell_bold),
            Paragraph("284,807 Transactions<br/>0.172% Fraud (578:1)", table_cell_style),
            Paragraph("Retained strictly as a numerical ablation benchmark to mathematically verify that our class-conditional calibration and MC Dropout formulations hold under extreme 578:1 skew.", table_cell_style)
        ]
    ]

    data_table = Table(data_table_data, colWidths=[130, 94, 280])
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('GRID', (0,0), (-1,-1), 0.6, CARD_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(data_table)

    # =========================================================
    # PAGE 4: EXPERIMENTS, IMPACT, ROADMAP & TEAM ROLES
    # =========================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Experimental Design, Baselines & Target Metrics", h1_style))
    p_exp = (
        "We establish rigorous benchmarking across four model families:<br/>"
        "• <b>Tree Ensembles:</b> XGBoost, LightGBM, CatBoost (Grinsztajn et al., NeurIPS 2022).<br/>"
        "• <b>Deep Tabular SOTA:</b> Uncalibrated MLP, FT-Transformer (Gorishniy et al., NeurIPS 2021).<br/>"
        "• <b>Standalone Tabular LLMs:</b> Zero-Shot TabLLM (Hegselmann et al., AISTATS 2023) and Full-Stream FinFRE-RAG (Tan et al., ACL 2026).<br/>"
        "• <b>Proposed Hybrid Cascade:</b> Calibrated Tabular System-1 + Dual Uncertainty Gate + System-2 RAG-LLM Auditor.<br/><br/>"
        "<b>Comprehensive Multi-Metric Evaluation:</b><br/>"
        "1. <i>Discrimination:</i> Precision-Recall AUC (PR-AUC / Average Precision), ROC-AUC, Recall @ 1% False Positive Rate (Feedzai industrial standard).<br/>"
        "2. <i>Reliability:</i> Binary ECE, Class-Conditional ECE (ECE_{y=1}, ECE_{y=0}), and Brier Score loss.<br/>"
        "3. <i>Operational Efficiency:</i> Area Under Risk-Coverage (AURC), Dollar Financial Loss ($), and Mean Latency per Transaction (ms/tx)."
    )
    story.append(Paragraph(p_exp, body_style))

    story.append(Paragraph("7. Practical Industry Impact & Regulatory Alignment (EU AI Act)", h1_style))
    p_impact = (
        "<b>Economic Viability for Global Payment Gateways:</b> Banks cannot deploy LLMs on 10,000 tx/sec streams without collapsing their infrastructure. "
        "Our uncertainty gate reduces LLM invocations by ~90%, making LLM fraud auditing deployable in real-world banking production.<br/><br/>"
        "<b>Compliance with High-Risk AI Legislation (EU AI Act &amp; Fed SR 11-7):</b> The EU AI Act strictly mandates that high-risk financial AI systems "
        "provide mathematical risk bounds, human oversight mechanisms, and interpretable audit logs. When our system defers an edge case, the System-2 LLM "
        "generates a natural language explanation citing exact behavioral risk factors (e.g., 'Unusual distance of 420km from home paired with a 4th transaction within 1 hour').<br/><br/>"
        "<b>Fairness &amp; Customer Protection:</b> Standard overconfident models frequently trigger wrongful card freezes on legitimate cardholders traveling or making atypical purchases. "
        "Our framework identifies unfamiliarity as high epistemic uncertainty rather than instant fraud, routing the transaction to contextual LLM/human review and reducing insulting card declines."
    )
    story.append(Paragraph(p_impact, body_style))

    story.append(Paragraph("8. Team Roles, Responsibilities & 4-Week Milestone Roadmap", h1_style))
    team_table_data = [
        [
            Paragraph("<b>Member</b>", table_header_style),
            Paragraph("<b>Technical Domain</b>", table_header_style),
            Paragraph("<b>Key Responsibilities &amp; Deliverables</b>", table_header_style)
        ],
        [
            Paragraph("<b>Hammad</b><br/>(Lead / Presenter)", table_cell_bold),
            Paragraph("Uncertainty Quantification &amp; Architecture", table_cell_style),
            Paragraph("Leads overall presentation, oversees MC Dropout epistemic variance engine (T=50 passes), parallel tensor batching, and System-1 to System-2 pipeline orchestration.", table_cell_style)
        ],
        [
            Paragraph("<b>Ammar</b>", table_cell_bold),
            Paragraph("Calibration &amp; Conformal Guarantees", table_cell_style),
            Paragraph("Implements Temperature Scaling (T*), Class-Conditional ECE metrics (ECE_{y=1}, ECE_{y=0}), and finite-sample Split Conformal Prediction bounds.", table_cell_style)
        ],
        [
            Paragraph("<b>Daud</b>", table_cell_bold),
            Paragraph("Data Engineering &amp; Drift Protocols", table_cell_style),
            Paragraph("Manages the 2024 calendar-year and IEEE-CIS dataset pipelines, temporal chronological splitting, target encoding, and leakage-free feature scaling.", table_cell_style)
        ],
        [
            Paragraph("<b>Muhammad Rizwan Kareem</b>", table_cell_bold),
            Paragraph("LLM Auditing &amp; Serialization", table_cell_style),
            Paragraph("Engineers the Table-to-Text serialization module, FinFRE-RAG exemplar retrieval from training memory, and LLM prompt optimization for Tier-2 resolution.", table_cell_style)
        ],
        [
            Paragraph("<b>Sarmad</b>", table_cell_bold),
            Paragraph("Selective Policy &amp; Cost Modeling", table_cell_style),
            Paragraph("Formulates the financial cost matrix optimization (C_FN, C_FP, C_audit), sweeps Risk-Coverage curves, and benchmarks against tree baselines (XGBoost/LightGBM).", table_cell_style)
        ]
    ]

    team_table = Table(team_table_data, colWidths=[110, 114, 280])
    team_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.6, CARD_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(team_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>4-Week Course Execution Roadmap:</b>", h2_style))
    roadmap_text = (
        "• <b>30 Oct:</b> Individual Literature Reviews (10–15 papers per member comparing tabular SOTA, calibration, and LLMs).<br/>"
        "• <b>31 Oct:</b> Refined Idea Presentation (5-slide group deck incorporating the 2024 dataset and System-1/2 cascade).<br/>"
        "• <b>01–07 Nov:</b> Data pipeline ingestion for the 2024 dataset; train System-1 and GBDT baselines across temporal splits.<br/>"
        "• <b>08–15 Nov:</b> Implement Table-to-Text RAG-LLM module on abstained test subset; submit Short Progress Paper (15 Nov).<br/>"
        "• <b>16–22 Nov:</b> Run ablation studies and complete Full Project Presentation (22 Nov).<br/>"
        "• <b>23–30 Nov:</b> Finalize and format camera-ready research paper for CORE A/A* conference submission (30 Nov)."
    )
    story.append(Paragraph(roadmap_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Successfully compiled {filename}!")

if __name__ == "__main__":
    build_pdf()
