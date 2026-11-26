import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Academic Presentation Palette
    C_BG = RGBColor(248, 250, 252)          # Slate 50
    C_CARD = RGBColor(255, 255, 255)        # Pure White
    C_BORDER = RGBColor(226, 232, 240)      # Slate 200
    
    C_NAVY = RGBColor(15, 23, 42)           # Slate 900
    C_BLUE = RGBColor(30, 64, 175)          # Blue 800
    C_BLUE_MED = RGBColor(37, 99, 235)      # Blue 600
    C_BLUE_LIGHT = RGBColor(239, 246, 255)  # Blue 50
    
    C_RED = RGBColor(220, 38, 38)           # Red 600
    C_RED_LIGHT = RGBColor(254, 242, 242)   # Red 50
    
    C_GREEN = RGBColor(5, 150, 105)         # Emerald 600
    C_GREEN_MED = RGBColor(16, 185, 129)    # Emerald 500
    C_GREEN_LIGHT = RGBColor(236, 253, 245) # Emerald 50
    
    C_GOLD = RGBColor(217, 119, 6)          # Amber 600
    C_GOLD_LIGHT = RGBColor(254, 252, 232)  # Amber 50
    
    C_TEXT_MAIN = RGBColor(30, 41, 59)      # Slate 800
    C_TEXT_MUTED = RGBColor(100, 116, 139)  # Slate 500

    def add_header(slide, slide_num, title, subtitle):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.32), Inches(11.733), Inches(1.05))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p = tf.paragraphs[0]
        run_pill = p.add_run()
        run_pill.text = f"SLIDE {slide_num} OF 5  |  LUMS MS AI RESEARCH IDEA PRESENTATION  |  PRESENTER: MUHAMMAD RIZWAN KAREEM\n"
        run_pill.font.size = Pt(9.5)
        run_pill.font.bold = True
        run_pill.font.color.rgb = C_BLUE_MED
        
        run_title = p.add_run()
        run_title.text = title
        run_title.font.size = Pt(20)
        run_title.font.bold = True
        run_title.font.color.rgb = C_NAVY
        
        if subtitle:
            p2 = tf.add_paragraph()
            p2.space_before = Pt(3)
            run_sub = p2.add_run()
            run_sub.text = subtitle
            run_sub.font.size = Pt(11)
            run_sub.font.color.rgb = C_TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=C_CARD, border_color=C_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.1)
        else:
            shape.line.fill.background()
        return shape

    # =========================================================================
    # SLIDE 1: MOTIVATION AND LITERATURE GAP
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    add_header(
        slide1, 1, 
        "Motivation & Research Gap: The Reliability Crisis in Fraud Detection", 
        "Why high-stakes financial classification requires moving beyond raw discriminative metrics"
    )

    # Left Column: Operational Problem Card (Height 3.9 in)
    add_card(slide1, Inches(0.8), Inches(1.5), Inches(5.6), Inches(3.9))
    box1_l = slide1.shapes.add_textbox(Inches(1.0), Inches(1.62), Inches(5.2), Inches(3.6))
    tf1_l = box1_l.text_frame; tf1_l.word_wrap = True
    p = tf1_l.paragraphs[0]; p.text = "🎯 Operational Motivation & Problem Definition"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_BLUE

    points_l = [
        ("High-Stakes Financial Dilemma: ", "Credit card fraud costs global institutions >$32B annually. Operations face severe cost asymmetry: missed fraud causes full financial liability ($250+), while false alarms cause customer churn ($15)."),
        ("The Illusion of ROC-AUC: ", "Existing fraud classifiers routinely report ROC-AUC > 0.99. However, high ranking discrimination does NOT mean predicted probabilities are reliable or calibrated for automated decision-making."),
        ("Miscalibration Under Extreme Skew: ", "Transactions exhibit ~0.17% fraud prevalence. Under such extreme imbalance, standard neural networks produce distorted, overconfident probabilities on both classes."),
        ("The Danger of Overconfident Errors: ", "Standard point-estimate models conflate confidence with correctness. When an algorithm is wrong with >90% subjective confidence, it bypasses security checks undetected.")
    ]
    for title, text in points_l:
        p = tf1_l.add_paragraph(); p.space_before = Pt(6)
        r_t = p.add_run(); r_t.text = "• " + title; r_t.font.bold = True; r_t.font.size = Pt(9.8); r_t.font.color.rgb = C_NAVY
        r_b = p.add_run(); r_b.text = text; r_b.font.size = Pt(9.2); r_b.font.color.rgb = C_TEXT_MAIN

    # Bottom Callout Card on Left: Core Research Question (Height 1.55 in)
    add_card(slide1, Inches(0.8), Inches(5.55), Inches(5.6), Inches(1.55), bg_color=C_BLUE_LIGHT, border_color=C_BLUE_MED)
    box1_alert = slide1.shapes.add_textbox(Inches(1.0), Inches(5.65), Inches(5.2), Inches(1.35))
    tf1_alt = box1_alert.text_frame; tf1_alt.word_wrap = True
    p_alt = tf1_alt.paragraphs[0]
    r_alt1 = p_alt.add_run(); r_alt1.text = "💡 Core Research Question:\n"; r_alt1.font.bold = True; r_alt1.font.size = Pt(10.5); r_alt1.font.color.rgb = C_BLUE
    r_alt2 = p_alt.add_run(); r_alt2.text = "\"How can we design a trust-aware machine learning framework that calibrates probability estimates, quantifies epistemic uncertainty, and selectively routes high-risk transactions to human fraud investigators?\""; r_alt2.font.size = Pt(9.2); r_alt2.font.color.rgb = C_NAVY

    # Right Column: 3 Literature Gaps as Distinct Cards (Height 1.72 in each)
    gap_card_w = Inches(5.8)
    gap_card_h = Inches(1.72)
    gap_top = Inches(1.5)

    gaps_info = [
        ("Gap 1: Neglected Calibration Under Extreme Skew (Guo et al. ICML 2017)",
         "While modern deep nets are proven to be miscalibrated, existing calibration literature focuses predominantly on balanced vision/NLP benchmarks, leaving extreme imbalanced tabular financial data under-investigated.",
         C_RED),
        ("Gap 2: Point-Estimate Uncertainty Blindness (Gal & Ghahramani ICML 2016)",
         "Deterministic tabular models cannot distinguish between aleatoric ambiguity (inherent feature overlap near the boundary) and epistemic ignorance (model unfamiliarity with novel, zero-day fraud syndicate patterns).",
         C_GOLD),
        ("Gap 3: Missing Risk-Coverage Selective Triage (Geifman & El-Yaniv NeurIPS 2017)",
         "Current commercial fraud systems enforce rigid binary accept/reject decisions. They lack principled abstention mechanisms to balance the risk-coverage trade-off and optimize human-in-the-loop workflows.",
         C_BLUE)
    ]

    for i, (g_title, g_desc, g_color) in enumerate(gaps_info):
        y_pos = gap_top + i * Inches(1.92)
        add_card(slide1, Inches(6.7), y_pos, gap_card_w, gap_card_h, bg_color=C_CARD, border_color=g_color)
        b_box = slide1.shapes.add_textbox(Inches(6.88), y_pos + Inches(0.12), Inches(5.45), Inches(1.48))
        tf_g = b_box.text_frame; tf_g.word_wrap = True
        p_g = tf_g.paragraphs[0]
        r_gt = p_g.add_run(); r_gt.text = "📌 " + g_title + "\n"; r_gt.font.bold = True; r_gt.font.size = Pt(10.2); r_gt.font.color.rgb = g_color
        r_gd = p_g.add_run(); r_gd.text = g_desc; r_gd.font.size = Pt(9.2); r_gd.font.color.rgb = C_TEXT_MAIN

    # =========================================================================
    # SLIDE 2: PROPOSED SOLUTION AND NOVELTY
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(
        slide2, 2,
        "Proposed Solution & Novelty: Dual-Gated Trust-Aware Decision Framework",
        "A holistic proposed architecture unifying Post-Hoc Calibration, Epistemic Uncertainty, and Selective Triage"
    )

    card_w2 = Inches(3.7)
    card_h2 = Inches(4.25)
    card_y2 = Inches(1.5)

    # Pillar 1: Proposed Calibration
    add_card(slide2, Inches(0.8), card_y2, card_w2, card_h2)
    box2_1 = slide2.shapes.add_textbox(Inches(0.95), card_y2 + Inches(0.12), Inches(3.4), Inches(3.1))
    tf2_1 = box2_1.text_frame; tf2_1.word_wrap = True
    p = tf2_1.paragraphs[0]; p.text = "1. Post-Hoc Calibration"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_BLUE
    p_sub = tf2_1.add_paragraph(); p_sub.text = "Proposed Framework (Guo et al. ICML 2017)"; p_sub.font.size = Pt(9); p_sub.font.color.rgb = C_TEXT_MUTED

    p1_items = [
        ("Temperature Scaling (T*): ", "We propose optimizing a single validation temperature parameter on logits via L-BFGS to re-align output probabilities with true empirical rates without altering ROC ranking."),
        ("Class-Conditional ECE: ", "We propose evaluating Expected Calibration Error separately for Fraud (ECE_y=1) and Legitimate (ECE_y=0) classes to protect against hidden minority miscalibration."),
        ("Brier Loss Decomposition: ", "Tracks reliability and resolution to ensure sharper, trustworthy probabilistic scoring for downstream credit limits.")
    ]
    for title, text in p1_items:
        p = tf2_1.add_paragraph(); p.space_before = Pt(5)
        r1 = p.add_run(); r1.text = "• " + title; r1.font.bold = True; r1.font.size = Pt(9.2); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(8.8); r2.font.color.rgb = C_TEXT_MAIN

    # Equation box 1
    add_card(slide2, Inches(0.95), card_y2 + Inches(3.3), Inches(3.4), Inches(0.8), bg_color=C_BLUE_LIGHT, border_color=C_BLUE)
    eq1_box = slide2.shapes.add_textbox(Inches(1.05), card_y2 + Inches(3.35), Inches(3.2), Inches(0.7))
    tf_eq1 = eq1_box.text_frame; tf_eq1.word_wrap = True
    p_e1 = tf_eq1.paragraphs[0]
    p_e1.text = "Proposed: p̂ = σ(z / T*)\nOptimization: min_T NLL(z_val / T, y_val)"
    p_e1.font.size = Pt(8.5); p_e1.font.bold = True; p_e1.font.color.rgb = C_BLUE

    # Pillar 2: Proposed Epistemic Uncertainty
    add_card(slide2, Inches(4.8), card_y2, card_w2, card_h2)
    box2_2 = slide2.shapes.add_textbox(Inches(4.95), card_y2 + Inches(0.12), Inches(3.4), Inches(3.1))
    tf2_2 = box2_2.text_frame; tf2_2.word_wrap = True
    p = tf2_2.paragraphs[0]; p.text = "2. Bayesian Epistemic Engine"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_GOLD
    p_sub = tf2_2.add_paragraph(); p_sub.text = "Proposed Framework (Gal & Ghahramani ICML 2016)"; p_sub.font.size = Pt(9); p_sub.font.color.rgb = C_TEXT_MUTED

    p2_items = [
        ("Monte Carlo Dropout (T=50): ", "We propose maintaining active dropout (p=0.3) at test time across 50 stochastic passes to approximate deep Gaussian processes."),
        ("Epistemic Std (σ_MC): ", "Measures model ignorance: transactions with high σ_MC represent unfamiliar fraud vectors or unseen distributions."),
        ("Disentangling Uncertainty: ", "Separates aleatoric risk (inherently noisy overlap near boundary) from epistemic risk (novel fraud zero-day attacks).")
    ]
    for title, text in p2_items:
        p = tf2_2.add_paragraph(); p.space_before = Pt(5)
        r1 = p.add_run(); r1.text = "• " + title; r1.font.bold = True; r1.font.size = Pt(9.2); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(8.8); r2.font.color.rgb = C_TEXT_MAIN

    # Equation box 2
    add_card(slide2, Inches(4.95), card_y2 + Inches(3.3), Inches(3.4), Inches(0.8), bg_color=C_GOLD_LIGHT, border_color=C_GOLD)
    eq2_box = slide2.shapes.add_textbox(Inches(5.05), card_y2 + Inches(3.35), Inches(3.2), Inches(0.7))
    tf_eq2 = eq2_box.text_frame; tf_eq2.word_wrap = True
    p_e2 = tf_eq2.paragraphs[0]
    p_e2.text = "Predictive Mean: p̄ = (1/T) ∑ σ(z_t)\nEpistemic Std: σ_MC = √[ (1/T) ∑ (p_t - p̄)² ]"
    p_e2.font.size = Pt(8.5); p_e2.font.bold = True; p_e2.font.color.rgb = C_GOLD

    # Pillar 3: Proposed Selective Classification
    add_card(slide2, Inches(8.8), card_y2, card_w2, card_h2)
    box2_3 = slide2.shapes.add_textbox(Inches(8.95), card_y2 + Inches(0.12), Inches(3.4), Inches(3.1))
    tf2_3 = box2_3.text_frame; tf2_3.word_wrap = True
    p = tf2_3.paragraphs[0]; p.text = "3. Dual-Gated Selective Triage"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_GREEN
    p_sub = tf2_3.add_paragraph(); p_sub.text = "Proposed Framework (Geifman & El-Yaniv NeurIPS 2017)"; p_sub.font.size = Pt(9); p_sub.font.color.rgb = C_TEXT_MUTED

    p3_items = [
        ("Dual-Gated Rejection Rule: ", "We propose automating a transaction IF AND ONLY IF:\n  1. Confidence ≥ τ_conf (Aleatoric Gate)\n  2. σ_epistemic ≤ τ_σ (Epistemic Gate)"),
        ("Tier-2 Human-in-the-Loop: ", "Abstained / high-uncertainty transactions are automatically routed to human fraud experts, aiming to eliminate silent false negatives."),
        ("Cost-Sensitive Objective: ", "Formulates decision thresholds to minimize financial loss: Total Loss = C_FN · FN + C_FP · FP + C_audit · Audit.")
    ]
    for title, text in p3_items:
        p = tf2_3.add_paragraph(); p.space_before = Pt(5)
        r1 = p.add_run(); r1.text = "• " + title; r1.font.bold = True; r1.font.size = Pt(9.2); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(8.8); r2.font.color.rgb = C_TEXT_MAIN

    # Equation box 3
    add_card(slide2, Inches(8.95), card_y2 + Inches(3.3), Inches(3.4), Inches(0.8), bg_color=C_GREEN_LIGHT, border_color=C_GREEN)
    eq3_box = slide2.shapes.add_textbox(Inches(9.05), card_y2 + Inches(3.35), Inches(3.2), Inches(0.7))
    tf_eq3 = eq3_box.text_frame; tf_eq3.word_wrap = True
    p_e3 = tf_eq3.paragraphs[0]
    p_e3.text = "Automate IF: Conf ≥ τ_conf AND σ_MC ≤ τ_σ\nOtherwise: Defer to Human Tier-2 Audit"
    p_e3.font.size = Pt(8.5); p_e3.font.bold = True; p_e3.font.color.rgb = C_GREEN

    # Bottom Proposed Flow Banner
    add_card(slide2, Inches(0.8), Inches(5.9), Inches(11.733), Inches(1.2), bg_color=C_BLUE_LIGHT, border_color=C_BLUE_MED)
    box2_flow = slide2.shapes.add_textbox(Inches(0.95), Inches(5.98), Inches(11.4), Inches(1.05))
    tf2_f = box2_flow.text_frame; tf2_f.word_wrap = True
    p_f = tf2_f.paragraphs[0]
    r_f1 = p_f.add_run(); r_f1.text = "🔄 Proposed Decision Pipeline & Key Novelty:\n"; r_f1.font.bold = True; r_f1.font.size = Pt(10.5); r_f1.font.color.rgb = C_BLUE
    r_f2 = p_f.add_run(); r_f2.text = "Transaction x  ➔  CreditNet Logits  ➔  [Gate 1: Calibrated Temp Scaling T*]  ➔  [Gate 2: MC Dropout (T=50) σ_MC]  ➔  Dual-Gated Policy:\n"; r_f2.font.size = Pt(9.5); r_f2.font.color.rgb = C_NAVY
    r_f3 = p_f.add_run(); r_f3.text = "  • High Trust (85-90% volume): Automated Execution  |  • Low Trust / High Ignorance: Routed to Tier-2 Human Fraud Auditor"; r_f3.font.size = Pt(9.5); r_f3.font.bold = True; r_f3.font.color.rgb = C_GREEN

    # =========================================================================
    # SLIDE 3: DATASET AND METHODOLOGY
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(
        slide3, 3,
        "Dataset & Methodology: Planned Experimental Design & Technical Pipeline",
        "Planned benchmark datasets, leakage-free 3-way evaluation protocol, and neural architecture"
    )

    # Left Column: Datasets & Protocol Card
    add_card(slide3, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.6))
    box3_l = slide3.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.2), Inches(5.3))
    tf3_l = box3_l.text_frame; tf3_l.word_wrap = True
    p = tf3_l.paragraphs[0]; p.text = "📁 Planned Benchmark Datasets & Protocol"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_BLUE

    d_items = [
        ("Primary Dataset (European Cardholders): ", "284,807 real transactions with 492 fraud cases (0.172% positive class). 30 numerical features: 28 PCA components (V1-V28) + Time + Amount."),
        ("Auxiliary Cross-Domain Benchmark: ", "Credit Risk Assessment dataset (demographics, income, debt payments, credit limits) to validate cross-domain generalizability."),
        ("Leakage-Free 3-Way Split: ", "Train (60%), Validation (20%), and Test (20%). Scaler parameters (mean, std) will be strictly fitted on Train and applied forward to Val/Test."),
        ("Temporal Chronological Splitting: ", "In addition to stratified splitting, chronological splits along the 'Time' dimension will be evaluated to measure robustness against concept drift."),
        ("Class Imbalance Handling: ", "Dynamic positive class weight w_pos = N_neg / N_pos ≈ 578.1 integrated into BCEWithLogitsLoss during backprop to avoid majority class bias.")
    ]
    for title, text in d_items:
        p = tf3_l.add_paragraph(); p.space_before = Pt(8)
        r1 = p.add_run(); r1.text = "• " + title; r1.font.bold = True; r1.font.size = Pt(9.8); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(9.2); r2.font.color.rgb = C_TEXT_MAIN

    # Right Column: Proposed Neural Pipeline
    add_card(slide3, Inches(6.7), Inches(1.5), Inches(5.8), Inches(5.6))
    box3_r = slide3.shapes.add_textbox(Inches(6.9), Inches(1.65), Inches(5.4), Inches(5.3))
    tf3_r = box3_r.text_frame; tf3_r.word_wrap = True
    p = tf3_r.paragraphs[0]; p.text = "⚙️ Proposed Neural Pipeline & Architecture"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_NAVY

    m_items = [
        ("1. CreditNet Backbone Architecture: ", "Multi-layer perceptron (Linear[30➔64] ➔ ReLU ➔ Dropout[0.3] ➔ Linear[64➔32] ➔ ReLU ➔ Dropout[0.3] ➔ Linear[32➔1]). Dropout layers remain active during test-time sampling."),
        ("2. Training Configuration: ", "Adam optimizer (lr=1e-3, weight_decay=1e-5), mini-batch size 512, early stopping monitored on validation loss."),
        ("3. Temperature Scaling Optimizer: ", "Extracts validation logits and solves scalar optimization min_T NLL(z_val / T, y_val) via L-BFGS-B (Guo et al. 2017)."),
        ("4. Epistemic Uncertainty Engine: ", "Performs T=50 stochastic forward passes per test transaction, calculating predictive mean p_mean and epistemic standard deviation σ_MC."),
        ("5. Cost-Sensitive Selective Policy: ", "Evaluates transactions under financial loss matrix (C_FN=$250, C_FP=$15, C_audit=$5) to plot Risk-Coverage and find optimal operating threshold.")
    ]
    for title, text in m_items:
        p = tf3_r.add_paragraph(); p.space_before = Pt(7)
        r1 = p.add_run(); r1.text = title; r1.font.bold = True; r1.font.size = Pt(9.8); r1.font.color.rgb = C_BLUE
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(9.2); r2.font.color.rgb = C_TEXT_MAIN

    # =========================================================================
    # SLIDE 4: PROBABLE RESULTS
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(
        slide4, 4,
        "Probable Results: Expected Outcomes, Metrics & Baseline Benchmarking",
        "Hypothesized performance, targeted evaluation metrics, and anticipated comparison against baselines"
    )

    # Left Column: Hypothesized Outcomes & Evaluation Framework
    add_card(slide4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.6))
    box4_l = slide4.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.2), Inches(5.3))
    tf4_l = box4_l.text_frame; tf4_l.word_wrap = True
    p = tf4_l.paragraphs[0]; p.text = "📊 Hypothesized Outcomes & Target Metrics"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_GREEN

    res_items = [
        ("Established Baselines for Benchmarking: ", "We will compare the proposed approach against: (1) Standard Point-Estimate MLP, (2) Gradient Boosted Trees (XGBoost / LightGBM), and (3) Random Forests."),
        ("Preserved High Discrimination: ", "Targeting ROC-AUC > 0.99 and PR-AUC > 0.60, hypothesizing that calibration and selective abstention will maintain or enhance discriminative power."),
        ("Expected 40-50% ECE Error Reduction: ", "Anticipate that Temperature Scaling will decrease Expected Calibration Error from ~0.04 down to ~0.02, successfully eliminating the probability distortion caused by class skew."),
        ("Distinct Epistemic Uncertainty Gap: ", "Hypothesize that mean epistemic std σ_MC on misclassified samples will be significantly higher (+30-40%) than on correct samples, confirming σ_MC as an effective failure predictor."),
        ("Elimination of Silent Killer Errors: ", "Expect the dual-gated policy to intercept high-confidence false negatives and defer them to human review, preventing automated fraud leakage.")
    ]
    for title, text in res_items:
        p = tf4_l.add_paragraph(); p.space_before = Pt(8)
        r1 = p.add_run(); r1.text = "• " + title; r1.font.bold = True; r1.font.size = Pt(9.8); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(9.2); r2.font.color.rgb = C_TEXT_MAIN

    # Right Column: Visual Conceptual Illustration
    add_card(slide4, Inches(6.7), Inches(1.5), Inches(5.8), Inches(5.6))
    box4_r = slide4.shapes.add_textbox(Inches(6.9), Inches(1.65), Inches(5.4), Inches(0.45))
    tf4_r = box4_r.text_frame; p = tf4_r.paragraphs[0]; p.text = "📈 Conceptual Illustration of Anticipated Dynamics"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_NAVY

    hyp_path = "hypothesized_results.png"
    if os.path.exists(hyp_path):
        slide4.shapes.add_picture(hyp_path, Inches(6.85), Inches(2.25), width=Inches(5.5))

    cap_box = slide4.shapes.add_textbox(Inches(6.9), Inches(6.15), Inches(5.4), Inches(0.8))
    tf_c = cap_box.text_frame; tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = "Figure: Anticipated behavior. Left: Calibrated model is expected to track the ideal diagonal, correcting baseline overconfidence. Right: Selective classification is expected to drive selective risk to near-zero as coverage is dynamically tuned."
    p_c.font.size = Pt(8.5); p_c.font.color.rgb = C_TEXT_MUTED

    # =========================================================================
    # SLIDE 5: IMPACT AND FEASIBILITY
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(
        slide5, 5,
        "Impact & Feasibility: Practical Significance & Research Roadmap",
        "Delivering concrete financial savings, safe human-AI collaboration, and CORE A/A* conference publication"
    )

    card_w5 = Inches(3.7)
    card_h5 = Inches(5.6)
    card_y5 = Inches(1.5)

    # Col 1: Operational & Financial Impact
    add_card(slide5, Inches(0.8), card_y5, card_w5, card_h5)
    box5_1 = slide5.shapes.add_textbox(Inches(0.95), card_y5 + Inches(0.12), Inches(3.4), Inches(5.3))
    tf5_1 = box5_1.text_frame; tf5_1.word_wrap = True
    p = tf5_1.paragraphs[0]; p.text = "💼 Industry & Practical Significance"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_GREEN
    
    b1_items = [
        ("Financial Fraud Loss Prevention: ", "Eliminating high-confidence false negatives protects financial institutions from undetected fraud rings and automated chargeback liability."),
        ("Reduced Customer Friction: ", "Calibrated probabilities prevent erroneous transaction declines on genuine customers, safeguarding multi-million dollar customer lifetime value."),
        ("Pragmatic Human-in-the-Loop: ", "Automates >85-90% of routine transactions with verified high trust, routing only genuinely uncertain cases to human fraud investigators."),
        ("Operational Cost Efficiency: ", "Optimizes fraud analyst capacity by filtering out obvious decisions and focusing human attention where epistemic uncertainty is highest.")
    ]
    for title, text in b1_items:
        p = tf5_1.add_paragraph(); p.space_before = Pt(8)
        r1 = p.add_run(); r1.text = "• " + title; r1.font.bold = True; r1.font.size = Pt(9.6); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(9.1); r2.font.color.rgb = C_TEXT_MAIN

    # Col 2: Academic Contribution
    add_card(slide5, Inches(4.8), card_y5, card_w5, card_h5)
    box5_2 = slide5.shapes.add_textbox(Inches(4.95), card_y5 + Inches(0.12), Inches(3.4), Inches(5.3))
    tf5_2 = box5_2.text_frame; tf5_2.word_wrap = True
    p = tf5_2.paragraphs[0]; p.text = "🎓 Academic Contribution"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_BLUE

    b2_items = [
        ("Bridging Theory to Tabular ML: ", "Translates theoretical uncertainty advances (Guo 2017, Gal 2016, Geifman 2017)—predominantly evaluated on vision/NLP—into real-world tabular finance."),
        ("Class-Conditional Trust Metrics: ", "Formulates rigorous evaluation standards for calibration and selective risk in extreme class-imbalanced environments."),
        ("Targeting CORE A/A* Venues: ", "Methodology and experimental design structured for premier AI/ML conferences such as NeurIPS, ICML, AAAI, or ICLR."),
        ("Safety & Reproducibility: ", "Contributes fully reproducible, modular open-source library for trust-aware financial classification.")
    ]
    for title, text in b2_items:
        p = tf5_2.add_paragraph(); p.space_before = Pt(8)
        r1 = p.add_run(); r1.text = "• " + title; r1.font.bold = True; r1.font.size = Pt(9.6); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(9.1); r2.font.color.rgb = C_TEXT_MAIN

    # Col 3: Feasibility & Roadmap
    add_card(slide5, Inches(8.8), card_y5, card_w5, card_h5)
    box5_3 = slide5.shapes.add_textbox(Inches(8.95), card_y5 + Inches(0.12), Inches(3.4), Inches(5.3))
    tf5_3 = box5_3.text_frame; tf5_3.word_wrap = True
    p = tf5_3.paragraphs[0]; p.text = "⏱️ Feasibility & Semester Roadmap"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_GOLD

    b3_items = [
        ("High Technical Feasibility: ", "Lightweight neural architecture requires minimal compute (executes in minutes on standard CPU/GPU hardware); datasets and theoretical frameworks are well-defined."),
        ("No GPU Compute Bottlenecks: ", "Compact feature dimensions allow rapid iteration, calibration fitting, and MC Dropout sampling without expensive infrastructure."),
        ("Course Milestone Alignment: ", "• Oct 24: Idea Presentation (Proposed Idea)\n• Oct 30: Literature Review (10-15 Papers)\n• Oct 31: Refined Idea Presentation\n• Nov 15: Short Progress Paper\n• Nov 22: Full Project Presentation\n• Nov 30: Final Research Paper (CORE A/A*)"),
        ("Clear Risk Mitigation: ", "Primary risks (imbalance mitigation, threshold optimization, baseline evaluation) are accounted for in the experimental plan.")
    ]
    for title, text in b3_items:
        p = tf5_3.add_paragraph(); p.space_before = Pt(8)
        r1 = p.add_run(); r1.text = "• " + title; r1.font.bold = True; r1.font.size = Pt(9.6); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = text; r2.font.size = Pt(9.1); r2.font.color.rgb = C_TEXT_MAIN

    out_pptx = "trust_aware_ml_presentation.pptx"
    prs.save(out_pptx)
    print(f"✅ Successfully created proposal-framed {out_pptx}!")

if __name__ == "__main__":
    create_deck()
