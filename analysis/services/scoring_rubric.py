"""Scoring Rubric Definitions for RAG-based Sentiment and Fundamental Analysis.

This module provides detailed scoring rubrics (0-100 scale) to prevent
polarization bias and ensure granular scoring in sentiment analysis.
"""

# Sentiment Analysis Rubric (0-100) for Stock News
SENTIMENT_RUBRIC = """
**Sentiment Scoring Rubric (0-100 scale) for Stock News:**

**90-100 (Extremely Bullish):**
- Record-breaking profits, revenue, or market share
- Major breakthrough announcements (FDA approval, patent grants, strategic partnerships)
- Exceptional earnings beats (>20% above expectations)
- Upgrades from major analysts with strong price targets
- Acquisition offers or buyout rumors at premium prices
- Example: "Company XYZ reports Q4 revenue of $10B, beating estimates by 25% and achieving record profit margins."

**80-89 (Very Bullish):**
- Strong positive growth trends with minor caveats
- Earnings beats (10-20% above expectations)
- Positive guidance revisions upward
- New product launches with strong market reception
- Market expansion or successful entry into new markets
- Example: "Company XYZ exceeds Q3 expectations by 15%, raises full-year guidance, though margins slightly compressed."

**70-79 (Bullish):**
- Solid earnings growth meeting or slightly exceeding expectations
- Positive operational updates without major concerns
- Moderate analyst upgrades
- Steady market share gains
- Example: "Company XYZ reports 8% revenue growth, in line with expectations, with stable margins."

**60-69 (Moderately Positive):**
- Mixed results with slight positive bias
- Minor positive developments offset by concerns
- Neutral-to-positive guidance
- Example: "Company XYZ revenue up 3%, but expenses increased; management maintains cautious outlook."

**50-59 (Neutral):**
- Standard operational updates with no significant impact
- Mixed signals (some positive, some negative)
- Results meeting expectations exactly
- No material changes to outlook
- Example: "Company XYZ reports quarterly results in line with expectations, no major changes to guidance."

**40-49 (Moderately Negative):**
- Mixed results with slight negative bias
- Minor misses or concerns
- Cautious or uncertain guidance
- Example: "Company XYZ revenue down 2%, margins compressed slightly; management cites market headwinds."

**30-39 (Bearish):**
- Earnings misses (5-15% below expectations)
- Negative guidance revisions
- Market share losses or competitive pressures
- Operational challenges or cost overruns
- Example: "Company XYZ misses Q3 earnings by 10%, lowers full-year guidance citing supply chain issues."

**20-29 (Very Bearish):**
- Significant earnings misses (>15% below expectations)
- Major operational failures or product recalls
- Regulatory concerns or investigations
- Significant analyst downgrades
- Example: "Company XYZ reports Q4 loss, misses revenue by 20%, announces restructuring amid declining demand."

**10-19 (Extremely Bearish):**
- Severe financial distress signals
- Bankruptcy warnings or going concern issues
- Major lawsuits with significant financial impact
- Regulatory bans or severe penalties
- Example: "Company XYZ warns of potential bankruptcy, faces SEC investigation, and reports massive quarterly loss."

**0-9 (Catastrophic):**
- Bankruptcy filings or liquidation
- Fraud allegations or criminal investigations
- Complete business model failure
- Example: "Company XYZ files for Chapter 11 bankruptcy protection amid fraud investigation."

**Constraints:**
- Do NOT default to 0 or 100 unless the language is extremely emotional and specific
- Use the full 0-100 range - avoid clustering scores
- Consider both magnitude and certainty of the news impact
- Factor in market context and company size (same news may score differently for large vs small cap)
"""

# Fundamental Analysis Rubric (0-100) for Financial Reports and Company Documents
FUNDAMENTAL_RUBRIC = """
**Fundamental Scoring Rubric (0-100 scale) for Financial Reports:**

**90-100 (Exceptional Fundamentals):**
- Consistently strong profitability (ROE >20%, ROA >15%)
- Excellent financial health (current ratio >2.0, debt-to-equity <0.3)
- Strong cash flow generation and low debt
- Consistent revenue and earnings growth (>15% YoY)
- High-quality balance sheet with significant cash reserves
- Example: "Company XYZ shows ROE of 25%, debt-to-equity of 0.2, and 20% revenue growth with strong free cash flow."

**80-89 (Very Strong Fundamentals):**
- Strong profitability metrics (ROE 15-20%, ROA 10-15%)
- Good financial health (current ratio 1.5-2.0, debt-to-equity 0.3-0.5)
- Positive cash flow trends
- Solid growth (10-15% YoY)
- Example: "Company XYZ demonstrates ROE of 18%, manageable debt levels, and consistent 12% revenue growth."

**70-79 (Strong Fundamentals):**
- Good profitability (ROE 10-15%, ROA 5-10%)
- Adequate financial health (current ratio 1.2-1.5, debt-to-equity 0.5-0.7)
- Positive but moderate growth (5-10% YoY)
- Stable cash flow
- Example: "Company XYZ maintains ROE of 12%, reasonable leverage, and steady 7% growth trajectory."

**60-69 (Moderately Positive Fundamentals):**
- Acceptable profitability (ROE 5-10%, ROA 2-5%)
- Adequate liquidity (current ratio 1.0-1.2)
- Moderate growth or stability (0-5% YoY)
- Some concerns but manageable
- Example: "Company XYZ shows modest profitability, adequate liquidity, with flat revenue growth."

**50-59 (Neutral Fundamentals):**
- Average profitability metrics
- Balanced financial position
- Stable but no significant growth
- No major red flags or strengths
- Example: "Company XYZ reports average financial metrics with stable operations, no significant changes."

**40-49 (Moderately Weak Fundamentals):**
- Below-average profitability (ROE 2-5%, ROA 0-2%)
- Tight liquidity or increasing leverage
- Declining or negative growth trends
- Some financial concerns emerging
- Example: "Company XYZ shows declining profitability, increasing debt levels, and negative revenue growth."

**30-39 (Weak Fundamentals):**
- Poor profitability (ROE 0-2%, ROA negative or near zero)
- Financial stress indicators (current ratio <1.0, debt-to-equity >1.0)
- Negative growth trends
- Cash flow concerns
- Example: "Company XYZ reports low profitability, high leverage, and declining revenue with cash flow pressure."

**20-29 (Very Weak Fundamentals):**
- Loss-making or near break-even
- Significant financial distress (high debt, poor liquidity)
- Severe operational decline
- Going concern risks emerging
- Example: "Company XYZ shows losses, debt-to-equity >1.5, negative cash flow, and declining market position."

**10-19 (Critical Fundamentals):**
- Sustained losses
- Severe balance sheet problems
- Bankruptcy risk indicators
- Major operational failures
- Example: "Company XYZ reports significant losses, severe liquidity crisis, and potential going concern issues."

**0-9 (Fundamental Failure):**
- Bankruptcy or liquidation
- Complete financial collapse
- Fraud or major accounting irregularities
- Example: "Company XYZ files for bankruptcy with massive liabilities exceeding assets."

**Constraints:**
- Consider industry context (some industries have naturally different financial ratios)
- Factor in company lifecycle stage (growth companies may have different metrics)
- Do NOT default to extremes - use granular scoring
- Consider trend direction, not just absolute values
"""

# Few-shot examples for sentiment analysis
SENTIMENT_FEW_SHOT_EXAMPLES = """
**Example 1 (Score: 98 - Extremely Bullish):**
News: "TechGiant announces record-breaking Q4 revenue of $150B, exceeding estimates by 28%. Company achieves highest profit margins in history and announces $50B stock buyback program."
Analysis: Exceptional performance across all metrics - record revenue, massive beat, record margins, and shareholder-friendly action. This is truly exceptional news.
Score: 98

**Example 2 (Score: 95 - Extremely Bullish):**
News: "BiotechPharma receives FDA approval for breakthrough cancer treatment, expected to generate $2B in annual revenue."
Analysis: Major breakthrough (FDA approval) with significant revenue potential represents exceptional positive news.
Score: 95

**Example 3 (Score: 92 - Extremely Bullish):**
News: "MegaCorp announces acquisition offer at 40% premium to current stock price. Board recommends acceptance. Deal expected to close in Q2."
Analysis: Premium acquisition offer represents exceptional value creation for shareholders. Very positive development.
Score: 92

**Example 4 (Score: 88 - Very Bullish):**
News: "AutoMaker reports Q3 earnings beat of 22%, raises full-year guidance by 8%. New electric vehicle line exceeds sales targets by 30%."
Analysis: Strong beat with upward guidance revision and successful product launch. Very positive momentum.
Score: 88

**Example 5 (Score: 85 - Very Bullish):**
News: "RetailChain exceeds Q2 expectations by 16%, announces expansion into 50 new markets. Same-store sales up 12% YoY."
Analysis: Strong beat with aggressive expansion plans and solid same-store growth. Very bullish signals.
Score: 85

**Example 6 (Score: 82 - Very Bullish):**
News: "TechCorp reports Q3 revenue of $500M, beating analyst estimates by 18%. The company also raised full-year guidance by 5%."
Analysis: This shows strong positive growth (18% beat) with upward guidance revision, indicating confidence. However, it's not record-breaking.
Score: 82

**Example 7 (Score: 78 - Bullish):**
News: "PharmaCo reports solid Q1 results, beating estimates by 8%. New drug shows promising Phase 2 trial results. Management maintains positive outlook."
Analysis: Good beat with positive pipeline news. Solid performance but not exceptional.
Score: 78

**Example 8 (Score: 75 - Bullish):**
News: "IndustrialCorp reports earnings in line with expectations, revenue up 7% YoY. Announces new $500M contract win. Margins stable."
Analysis: Meeting expectations with growth and contract win. Positive but not exceptional.
Score: 75

**Example 9 (Score: 72 - Bullish):**
News: "ConsumerBrand reports Q4 revenue growth of 6%, slightly above expectations. New product line launches successfully. Guidance unchanged."
Analysis: Modest beat with successful product launch. Positive trend but conservative guidance.
Score: 72

**Example 10 (Score: 68 - Moderately Positive):**
News: "MediaCo reports mixed Q2: revenue up 4% but subscriber growth slowed. Management expects improvement in H2. No guidance change."
Analysis: Modest growth with some concerns about subscriber trends. Slightly positive overall.
Score: 68

**Example 11 (Score: 65 - Moderately Positive):**
News: "TechStartup reports revenue growth of 5%, but losses widened. Management cites investment in growth. Path to profitability extended by 6 months."
Analysis: Growth continues but profitability delayed. Mixed picture with slight positive bias.
Score: 65

**Example 12 (Score: 58 - Neutral):**
News: "RetailCo announces Q2 earnings in line with expectations. Revenue grew 3% YoY, margins remained stable."
Analysis: Results are neutral - meeting expectations with modest growth, no surprises positive or negative.
Score: 58

**Example 13 (Score: 55 - Neutral):**
News: "ManufacturingCo reports quarterly results exactly matching analyst estimates. No major operational changes. Guidance maintained."
Analysis: Perfectly neutral - meeting expectations exactly with no changes to outlook.
Score: 55

**Example 14 (Score: 52 - Neutral):**
News: "EnergyCo reports Q1 results: revenue flat, margins slightly down due to commodity price volatility. Management sees stability ahead."
Analysis: Flat results with minor margin pressure. Neutral with slight negative undertone.
Score: 52

**Example 15 (Score: 48 - Moderately Negative):**
News: "EnergyCorp reports mixed Q1 results: revenue up 5% but margins compressed due to rising costs. Management maintains cautious outlook."
Analysis: Mixed signals - positive revenue growth offset by margin pressure and cautious tone. Slight negative bias.
Score: 48

**Example 16 (Score: 45 - Moderately Negative):**
News: "Retailer reports Q3 revenue down 2%, misses earnings by 3%. Management cites competitive pressures but expects recovery."
Analysis: Minor miss with revenue decline. Negative but management maintains optimism.
Score: 45

**Example 17 (Score: 42 - Moderately Negative):**
News: "TechCo reports Q2 revenue growth slowed to 2%, margins compressed. Management lowers H2 guidance by 5% citing macro headwinds."
Analysis: Slowing growth with margin pressure and guidance reduction. Moderately negative.
Score: 42

**Example 18 (Score: 38 - Bearish):**
News: "AutoMaker misses Q4 earnings by 8%, cites supply chain issues. Management lowers full-year guidance by 10%."
Analysis: Meaningful miss with significant guidance reduction. Bearish signals.
Score: 38

**Example 19 (Score: 35 - Bearish):**
News: "ManufacturingInc misses Q4 earnings by 12%, citing supply chain disruptions. Management lowered next quarter guidance."
Analysis: Significant miss (12%) with negative forward guidance indicates operational challenges and reduced confidence.
Score: 35

**Example 20 (Score: 32 - Bearish):**
News: "RetailChain reports Q1 loss, revenue down 8%. Announces store closures and layoffs. Management cites structural challenges."
Analysis: Loss-making quarter with revenue decline and cost-cutting measures. Significant concerns.
Score: 32

**Example 21 (Score: 28 - Very Bearish):**
News: "TechStartup reports Q3 revenue miss of 18%, announces major restructuring. CEO steps down. Path to profitability unclear."
Analysis: Large miss with leadership change and restructuring. Very concerning developments.
Score: 28

**Example 22 (Score: 25 - Very Bearish):**
News: "PharmaCo reports Q2 loss, misses revenue by 20%, announces restructuring amid declining demand."
Analysis: Significant operational failure with large miss and restructuring. Very bearish.
Score: 25

**Example 23 (Score: 22 - Very Bearish):**
News: "EnergyCo reports massive Q4 loss, revenue down 25%. Announces going concern warning. Stock downgraded to sell by major analysts."
Analysis: Severe financial distress with going concern warning. Critical situation.
Score: 22

**Example 24 (Score: 18 - Extremely Bearish):**
News: "Retailer warns of potential bankruptcy, reports massive quarterly loss. Faces SEC investigation into accounting practices."
Analysis: Bankruptcy warning with regulatory investigation. Extremely bearish situation.
Score: 18

**Example 25 (Score: 15 - Extremely Bearish):**
News: "ManufacturingCo reports catastrophic Q3 results, warns of liquidity crisis. Major customer files for bankruptcy. Company exploring strategic alternatives."
Analysis: Multiple severe issues including liquidity crisis and customer bankruptcy. Extremely bearish.
Score: 15

**Example 26 (Score: 12 - Extremely Bearish):**
News: "TechCo faces delisting threat, reports fraud investigation. CEO arrested. Company warns of potential Chapter 11 filing."
Analysis: Fraud investigation with leadership crisis and bankruptcy risk. Catastrophic situation.
Score: 12

**Example 27 (Score: 8 - Catastrophic):**
News: "BiotechCo files for Chapter 11 bankruptcy protection amid fraud investigation. SEC charges company with securities fraud. Stock trading halted."
Analysis: Bankruptcy filing with fraud charges. Complete failure.
Score: 8

**Example 28 (Score: 5 - Catastrophic):**
News: "EnergyCorp announces liquidation, all assets to be sold. CEO charged with embezzlement. Company ceases operations immediately."
Analysis: Complete business failure with criminal charges. Catastrophic.
Score: 5
"""

# Few-shot examples for fundamental analysis
FUNDAMENTAL_FEW_SHOT_EXAMPLES = """
**Example 1 (Score: 97 - Exceptional Fundamentals):**
Report: "Company demonstrates exceptional financial strength: ROE of 28%, ROA of 18%, debt-to-equity of 0.15, current ratio of 2.5, and 25% revenue growth with $2B cash reserves."
Analysis: Outstanding metrics across all dimensions - exceptional profitability, minimal leverage, excellent liquidity, strong growth, and massive cash position.
Score: 97

**Example 2 (Score: 94 - Exceptional Fundamentals):**
Report: "Company shows ROE of 24%, debt-to-equity of 0.2, current ratio of 2.2, and 22% revenue growth with strong free cash flow generation and consistent margin expansion."
Analysis: Exceptional metrics across profitability, leverage, liquidity, and growth. All indicators are very strong with improving margins.
Score: 94

**Example 3 (Score: 92 - Exceptional Fundamentals):**
Report: "Company shows ROE of 22%, debt-to-equity of 0.25, current ratio of 2.1, and 18% revenue growth with strong free cash flow generation."
Analysis: Exceptional metrics across profitability, leverage, liquidity, and growth. All indicators are very strong.
Score: 92

**Example 4 (Score: 89 - Very Strong Fundamentals):**
Report: "Financials demonstrate ROE of 19%, ROA of 13%, debt-to-equity of 0.35, current ratio of 1.8, with 14% revenue growth and positive cash flow trends."
Analysis: Very strong fundamentals with excellent profitability, low leverage, good liquidity, and solid growth.
Score: 89

**Example 5 (Score: 86 - Very Strong Fundamentals):**
Report: "Company reports ROE of 17%, debt-to-equity of 0.4, current ratio of 1.7, with 12% revenue growth and consistent cash generation."
Analysis: Strong fundamentals across key metrics. Good profitability with manageable leverage and solid growth.
Score: 86

**Example 6 (Score: 83 - Very Strong Fundamentals):**
Report: "Company demonstrates ROE of 16%, manageable debt levels (0.45x equity), and consistent 11% revenue growth with improving margins."
Analysis: Very strong profitability and growth, with reasonable leverage. Positive trends.
Score: 83

**Example 7 (Score: 79 - Strong Fundamentals):**
Report: "Financials show ROE of 14%, debt-to-equity of 0.55, current ratio of 1.4, with 9% revenue growth and stable cash flow."
Analysis: Strong fundamentals with good profitability, acceptable leverage, and solid growth.
Score: 79

**Example 8 (Score: 76 - Strong Fundamentals):**
Report: "Company maintains ROE of 13%, debt-to-equity of 0.6, current ratio of 1.35, with 8% revenue growth and positive operating cash flow."
Analysis: Good fundamentals with solid profitability, reasonable leverage, and consistent growth.
Score: 76

**Example 9 (Score: 73 - Strong Fundamentals):**
Report: "Company reports ROE of 12%, reasonable leverage (0.65x equity), and steady 7% growth trajectory with adequate liquidity."
Analysis: Strong fundamentals with good profitability and growth, though leverage is moderate.
Score: 73

**Example 10 (Score: 69 - Moderately Positive Fundamentals):**
Report: "Financials show ROE of 9%, debt-to-equity of 0.7, current ratio of 1.2, with 5% revenue growth and stable operations."
Analysis: Moderately positive fundamentals - acceptable profitability and growth, though leverage is increasing.
Score: 69

**Example 11 (Score: 66 - Moderately Positive Fundamentals):**
Report: "Company shows ROE of 8%, debt-to-equity of 0.75, current ratio of 1.15, with 4% revenue growth. Some margin pressure but manageable."
Analysis: Acceptable fundamentals with modest growth. Some concerns but overall positive.
Score: 66

**Example 12 (Score: 62 - Moderately Positive Fundamentals):**
Report: "Strong profitability (ROE 16%) and good cash flow, but debt levels increased to 0.8x equity and liquidity tightened to 1.0."
Analysis: Mixed picture - good profitability offset by increasing leverage and tight liquidity. Moderately positive with concerns.
Score: 62

**Example 13 (Score: 58 - Neutral Fundamentals):**
Report: "Financial metrics are average: ROE of 7%, debt-to-equity of 0.5, current ratio of 1.1, with flat revenue growth."
Analysis: Neutral fundamentals - no major strengths or weaknesses. Average performance across key metrics.
Score: 58

**Example 14 (Score: 55 - Neutral Fundamentals):**
Report: "Company reports average financial metrics: ROE of 6%, debt-to-equity of 0.6, current ratio of 1.05, with 1% revenue growth."
Analysis: Neutral fundamentals - balanced metrics with minimal growth. No major concerns or strengths.
Score: 55

**Example 15 (Score: 52 - Neutral Fundamentals):**
Report: "Financials show ROE of 5.5%, debt-to-equity of 0.65, current ratio of 1.0, with revenue flat. Margins stable but growth absent."
Analysis: Neutral to slightly weak fundamentals - adequate but unimpressive metrics.
Score: 52

**Example 16 (Score: 48 - Moderately Weak Fundamentals):**
Report: "Company shows ROE of 4%, debt-to-equity increased to 0.85, current ratio of 0.95, with revenue declining 2%."
Analysis: Moderately weak fundamentals - low profitability, increasing leverage, tight liquidity, and declining revenue.
Score: 48

**Example 17 (Score: 45 - Moderately Weak Fundamentals):**
Report: "Financials indicate ROE of 3.5%, debt-to-equity of 0.9, current ratio of 0.92, with revenue down 3% and margin compression."
Analysis: Weak fundamentals with multiple concerns - low profitability, high leverage, poor liquidity, and declining revenue.
Score: 45

**Example 18 (Score: 42 - Moderately Weak Fundamentals):**
Report: "Company reports declining profitability (ROE 3%), increasing debt-to-equity to 1.0, current ratio of 0.9, with revenue down 4%."
Analysis: Moderately weak fundamentals with deteriorating metrics across the board.
Score: 42

**Example 19 (Score: 38 - Weak Fundamentals):**
Report: "Financials show ROE of 2%, debt-to-equity of 1.1, current ratio of 0.85, with revenue declining 6% and negative cash flow."
Analysis: Weak fundamentals - low profitability, high leverage, poor liquidity, declining revenue, and cash burn.
Score: 38

**Example 20 (Score: 35 - Weak Fundamentals):**
Report: "Company reports ROE of 1.5%, debt-to-equity of 1.2, current ratio of 0.8, with revenue down 8% and operating losses."
Analysis: Weak fundamentals with multiple red flags including low profitability, high leverage, and operating losses.
Score: 35

**Example 21 (Score: 32 - Weak Fundamentals):**
Report: "Financials indicate ROE near zero, debt-to-equity of 1.3, current ratio of 0.75, with revenue declining 10% and significant cash burn."
Analysis: Very weak fundamentals - minimal profitability, excessive leverage, poor liquidity, and severe cash burn.
Score: 32

**Example 22 (Score: 28 - Very Weak Fundamentals):**
Report: "Company reports ROE of 3%, increasing debt-to-equity to 1.2, current ratio of 0.9, with revenue declining 5% and negative cash flow."
Analysis: Multiple red flags: low profitability, high leverage, poor liquidity, declining revenue, and cash burn. Significant concerns.
Score: 28

**Example 23 (Score: 25 - Very Weak Fundamentals):**
Report: "Company shows operating losses, debt-to-equity of 1.4, current ratio of 0.7, with revenue down 12% and going concern warning issued."
Analysis: Very weak fundamentals with operating losses, excessive leverage, poor liquidity, and going concern risk.
Score: 25

**Example 24 (Score: 22 - Very Weak Fundamentals):**
Report: "Financials reveal sustained losses, debt-to-equity of 1.5, current ratio of 0.65, with revenue declining 15% and severe liquidity crisis."
Analysis: Critical financial situation with losses, excessive leverage, severe liquidity issues, and declining operations.
Score: 22

**Example 25 (Score: 18 - Critical Fundamentals):**
Report: "Company reports significant losses, debt-to-equity of 1.8, current ratio of 0.6, with revenue down 20%. Bankruptcy risk indicators present."
Analysis: Critical fundamentals - sustained losses, excessive leverage, severe liquidity crisis, and bankruptcy risk.
Score: 18

**Example 26 (Score: 15 - Critical Fundamentals):**
Report: "Financials show massive losses, debt-to-equity of 2.0, current ratio of 0.5, with revenue down 25%. Company exploring strategic alternatives including bankruptcy."
Analysis: Critical situation with massive losses, extreme leverage, severe liquidity crisis, and potential bankruptcy.
Score: 15

**Example 27 (Score: 12 - Critical Fundamentals):**
Report: "Company reports catastrophic losses, debt-to-equity of 2.5, current ratio of 0.4, with revenue down 30%. Going concern issues. Major accounting irregularities discovered."
Analysis: Critical failure with catastrophic losses, extreme leverage, severe liquidity crisis, and accounting issues.
Score: 12

**Example 28 (Score: 8 - Fundamental Failure):**
Report: "Company files for Chapter 11 bankruptcy. Financials reveal massive liabilities exceeding assets by 40%. Fraud investigation ongoing. Operations ceased."
Analysis: Complete fundamental failure - bankruptcy filing, negative equity, fraud investigation, and operational shutdown.
Score: 8

**Example 29 (Score: 5 - Fundamental Failure):**
Report: "Company announces liquidation. All assets to be sold. Financials show complete collapse with liabilities 3x assets. SEC investigation reveals massive fraud."
Analysis: Complete fundamental failure - liquidation, negative equity, and fraud. Total collapse.
Score: 5
"""
