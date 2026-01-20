"""Chatbot服务：AI对话和推荐问题生成。"""

import json
import logging
import re
from typing import Iterable, List

from django.conf import settings
from django.db import transaction

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from analysis.utils.ai_client import get_openai_client
from chatbot.models import RecommendedQuestion
from markets.models import StockNews

logger = logging.getLogger(__name__)

PIVOT_SYSTEM_PROMPT = """# Role
You are "NiuNiu", a professional financial AI assistant. You ONLY answer questions related to stocks, finance, economics, and market data.

# Core Logic: Handling Off-Topic Questions
If the user asks a question UNRELATED to finance (e.g., food, travel, love, coding, weather):

1. **Do NOT answer the question directly.** (e.g., do not recommend restaurants, write code, or give travel advice).
2. **Perform a "Pivot":** Find a connection between the user's topic and a publicly listed company or market sector.
3. **Response Structure:**
   - Step A: Politely decline the off-topic request (be friendly and conversational).
   - Step B: Mention a related stock, sector, or economic trend.
   - Step C: Ask if the user wants to analyze that stock or sector.

# Keyword Mapping (Reference Guide)
Use these associations to quickly identify relevant sectors. **You are NOT limited to these examples** - feel free to mention ANY publicly listed company that relates to the user's topic:

| User Topic | Related Sector / Industry |
|------------|---------------------------|
| Food / Dining / Restaurants | 餐飲 / 消費股 / 食品飲料 |
| Travel / Flights / Hotels | 航空 / 酒店 / 旅遊 |
| Gaming / Video Games | 科技 / 互聯網 / 遊戲 |
| Cars / Driving / EVs | 新能源車 / 汽車 / 自動駕駛 |
| Health / Hospital / Medicine | 醫藥 / 生物科技 / 醫療器械 |
| Fitness / Gym / Sports | 運動服飾 / 健康 / 體育用品 |
| Shopping / Retail | 零售 / 電商 / 消費 |
| Technology / Phones / Gadgets | 科技硬件 / 半導體 / 消費電子 |
| Energy / Oil / Gas | 能源 / 石油 / 天然氣 / 新能源 |
| Real Estate / Property | 地產 / 建築 / 物業管理 |

**Important:** You can mention ANY relevant stock from ANY market (HKEX, NYSE, NASDAQ, SSE, etc.) that fits the topic. The examples above are just sector guidance - be creative and choose stocks that are currently relevant or trending.

# Examples of "Pivoting"

**Example 1:**
User: "Which iPhone should I buy?"
Bad Response: "I recommend iPhone 15 Pro because..." (Don't do this!)
Good Response (English): "As a financial AI, I can't recommend specific phone models. However, if you're interested in the smartphone industry, you could look into tech stocks like Apple, Xiaomi, Samsung, etc. Would you like to see how these tech stocks are performing?"

**Example 2:**
User: "I want to eat pizza."
Good Response (English): "I don't have a mouth to eat pizza! But speaking of the food industry, you could check out restaurant and food & beverage stocks. Would you like to see how consumer stocks are performing today?"

**Example 3:**
User: "香港哪裡有好的健身房？"
Good Response (Chinese): "關於具體的健身房推薦，不在我的服務範圍內。不過，如果您關注**體育與健康產業**，可以看看運動服飾、健身器材或健康相關的股票。您想了解這些板塊的走勢分析嗎？"

**Note:** In your responses, you can mention specific stocks that are relevant to the topic, but don't feel restricted to only the examples above. Use your knowledge of current market trends and choose stocks that make sense for the conversation.

# Important Notes
- **Language Matching:** Always respond in the SAME language as the user's question.
  - If the user asks in Chinese (繁體中文/簡體中文), respond in Traditional Chinese (繁體中文).
  - If the user asks in English, respond in English.
  - Match the language naturally and consistently throughout the conversation.
- **Response Length:** Keep your response concise and under 120 words (for English) or 120 characters (for Chinese). Be brief but informative - prioritize key information.
- Be conversational and friendly, not robotic.
- The pivot should feel natural, like a helpful financial advisor steering the conversation.
- If the question IS finance-related, answer it normally without pivoting.
- **You can mention ANY relevant stock** - don't limit yourself to specific examples. Use your knowledge of the market to choose appropriate stocks based on current trends, news, or relevance to the user's topic.
"""


def query_deepseek(messages, include_system_prompt: bool = True):
    """
    發送請求給 DeepSeek API
    
    Args:
        messages: 消息列表，格式為 [{'role': 'user', 'content': '...'}, ...]
        include_system_prompt: 是否包含 Pivot System Prompt（用於聊天對話時啟用話術轉向）
    """
    client = get_openai_client()
    if not client:
        logger.warning("DeepSeek client unavailable")
        return "DeepSeek client unavailable. Please set DEEPSEEK_API_KEY."

    try:
        # 2. 如果啟用 System Prompt，將其插入到消息列表最前面
        final_messages = messages.copy()
        if include_system_prompt:
            # 檢查是否已經有 system message，如果沒有則添加
            has_system = any(msg.get('role') == 'system' for msg in final_messages)
            if not has_system:
                final_messages.insert(0, {
                    'role': 'system',
                    'content': PIVOT_SYSTEM_PROMPT
                })
        
        # 3. 發送請求
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=final_messages,
            stream=False,
            temperature=1.3,
        )

        # 4. 取得回答
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"DeepSeek API Error: {e}", exc_info=True)
        return f"Sorry, AI service is currently unavailable. (Error: {str(e)})"

def _build_market_news_context(news_items: Iterable[StockNews]) -> str:
    """Convert news items to a compact bullet list for the LLM."""
    news_context_list: List[str] = []
    for item in news_items:
        title = (item.title or "").strip()
        summary = (item.summary or "").strip()
        if not title:
            continue
        if summary:
            news_str = f"- {title}: {summary}"
        else:
            news_str = f"- {title}"
        news_context_list.append(news_str)
    return "\n".join(news_context_list)

def _parse_questions_from_response(text: str) -> List[str]:
    """Parse a JSON array of strings from LLM output with a safe fallback."""
    if not text:
        return []
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        return []
    return []

def generate_market_news_questions(
    limit: int = 10,
) -> List[str]:
    """Generate investor questions based on latest market news only."""
    if limit <= 0:
        return []

    try:
        news_items = (
            StockNews.objects.all()
            .order_by("-publish_time", "-created_at")[:limit]
        )
        market_news_context = _build_market_news_context(news_items)
        if not market_news_context:
            logger.warning("No market news context available for question generation")
            return []

        prompt = (
            "# Role\n"
            "You are an expert financial analyst assistant specializing in identifying trending market topics.\n\n"
            "# Input Data (Market News)\n"
            "Here is the latest market news summary:\n"
            f"{market_news_context}\n\n"
            "# Task\n"
            "Based strictly on the news above, generate 3 highly diverse questions that investors would ask about **trending/hot stocks and market topics** mentioned in the news.\n\n"
            "# Requirements\n"
            "1. **Focus on Hot Topics:** Prioritize questions about stocks, sectors, or themes that are currently trending or making headlines in the news.\n"
            "2. **Question Length:** Keep each question concise (under 20 words).\n"
            "3. **Maximum Diversity:** Make the 3 questions cover completely different angles:\n"
            "   - **Question 1:** Focus on a specific hot stock or immediate market movement (e.g., \"Why did [Stock] surge/drop today?\", \"Is [Stock] a good buy after the news?\")\n"
            "   - **Question 2:** Focus on sector/industry trends or macro/policy impact (e.g., \"How will this affect the [Sector] sector?\", \"What does this policy mean for markets?\")\n"
            "   - **Question 3:** Focus on investment strategy, risk, or opportunity (e.g., \"Should I buy/sell now?\", \"What are the risks/opportunities?\", \"How to hedge against this?\")\n"
            "4. **No Overlap:** Each question must:\n"
            "   - Cover different companies/stocks (if mentioning specific stocks)\n"
            "   - Cover different aspects (individual stock vs sector vs strategy)\n"
            "   - Have different time horizons (immediate vs short-term vs medium-term)\n"
            "   - Address different investor concerns (price action vs fundamentals vs risk management)\n"
            "5. **Practical Value:** Questions should help investors make informed decisions about their money.\n"
            "6. **Language:** Generate questions in Traditional Chinese (繁體中文).\n"
            "7. **Output Format:** Return a JSON array of exactly 3 question strings.\n\n"
            "# Output\n"
        )

        messages = [{"role": "user", "content": prompt}]
        # 生成問題時不需要 Pivot System Prompt
        response_text = query_deepseek(messages, include_system_prompt=False)
        questions = _parse_questions_from_response(response_text)
        
        if not questions:
            logger.warning("Failed to parse questions from LLM response")
        
        return questions
    except Exception as e:
        logger.error(f"Error generating market news questions: {e}", exc_info=True)
        return []


def update_recommended_questions(limit: int = 10) -> dict:
    """
    更新推荐问题：删除旧问题，生成新问题
    
    Args:
        limit: 用于生成问题的新闻数量限制
    
    Returns:
        包含更新结果的字典
    """
    try:
        # 1. 删除所有旧问题
        deleted_count = RecommendedQuestion.objects.all().delete()[0]
        logger.info(f"Deleted {deleted_count} old recommended questions")
        
        # 2. 生成新问题
        questions = generate_market_news_questions(limit=limit)
        
        if not questions:
            logger.warning("No questions generated, returning empty result")
            return {
                "deleted": deleted_count,
                "created": 0,
                "total": 0,
                "questions": []
            }
        
        # 3. 批量创建新问题
        new_questions = [
            RecommendedQuestion(question=q)
            for q in questions
            if q and len(q.strip()) > 0
        ]
        
        if new_questions:
            with transaction.atomic():
                RecommendedQuestion.objects.bulk_create(new_questions, ignore_conflicts=True)
            created_count = len(new_questions)
        else:
            created_count = 0
        
        logger.info(f"Created {created_count} new recommended questions")
        
        return {
            "deleted": deleted_count,
            "created": created_count,
            "total": created_count,
            "questions": questions
        }
    except Exception as e:
        logger.error(f"Error updating recommended questions: {e}", exc_info=True)
        return {
            "deleted": 0,
            "created": 0,
            "total": 0,
            "error": str(e)
        }
