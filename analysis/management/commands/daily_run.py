"""
Daily Run Management Command - 统一执行所有日常任务

运行方式：
    python manage.py daily_run
    python manage.py daily_run --skip-details --skip-news
"""

import logging
from datetime import datetime
from django.core.management.base import BaseCommand

from markets.services.update import update_stock_details, update_stock_news
from analysis.services.daily_tasks import run_efs_evolution, run_stock_analysis, update_rankings, update_market_context
from chatbot.services import update_recommended_questions

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run all daily tasks: update stock details, news, EFS evolution, recommended questions, market context, stock analysis, and rankings."

    def add_arguments(self, parser):
        parser.add_argument(
            '--skip-details',
            action='store_true',
            help='Skip updating stock details'
        )
        parser.add_argument(
            '--skip-news',
            action='store_true',
            help='Skip updating stock news'
        )
        parser.add_argument(
            '--skip-evolution',
            action='store_true',
            help='Skip EFS evolution'
        )
        parser.add_argument(
            '--skip-questions',
            action='store_true',
            help='Skip updating recommended questions'
        )
        parser.add_argument(
            '--skip-market-context',
            action='store_true',
            help='Skip updating market context'
        )
        parser.add_argument(
            '--skip-analysis',
            action='store_true',
            help='Skip stock analysis'
        )
        parser.add_argument(
            '--skip-rankings',
            action='store_true',
            help='Skip updating rankings'
        )

    def handle(self, *args, **options):
        start_time = datetime.now()
        self.stdout.write(self.style.SUCCESS("=" * 80))
        self.stdout.write(self.style.SUCCESS(f"Daily Run Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"))
        self.stdout.write(self.style.SUCCESS("=" * 80))
        
        results = {
            "start_time": start_time.isoformat(),
            "tasks": {}
        }

        # 1. Update Stock Details
        if not options.get('skip_details'):
            self.stdout.write(self.style.WARNING("\n[1/7] Updating Stock Details..."))
            try:
                details_result = update_stock_details()
                results["tasks"]["stock_details"] = details_result
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Success: {details_result['success']}/{details_result['total']} stocks"
                    )
                )
                if details_result['failed'] > 0:
                    self.stdout.write(
                        self.style.WARNING(f"  ✗ Failed: {details_result['failed']} stocks")
                    )
                    for error in details_result['errors']:
                        self.stdout.write(self.style.ERROR(f"    - {error}"))
                logger.info(f"Stock details updated: {details_result}")
            except Exception as e:
                error_msg = f"Error updating stock details: {str(e)}"
                results["tasks"]["stock_details"] = {"error": error_msg}
                self.stdout.write(self.style.ERROR(f"  ✗ {error_msg}"))
                logger.error(error_msg, exc_info=True)
        else:
            self.stdout.write(self.style.WARNING("\n[1/7] Skipping Stock Details Update"))
            results["tasks"]["stock_details"] = {"skipped": True}

        # 2. Update Stock News
        if not options.get('skip_news'):
            self.stdout.write(self.style.WARNING("\n[2/7] Updating Stock News..."))
            try:
                news_result = update_stock_news(days_back=3)
                results["tasks"]["stock_news"] = news_result
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Success: {news_result['success']}/{news_result['total_stocks']} stocks, "
                        f"{news_result['total_news']} news items"
                    )
                )
                if news_result['failed'] > 0:
                    self.stdout.write(
                        self.style.WARNING(f"  ✗ Failed: {news_result['failed']} stocks")
                    )
                    for error in news_result['errors']:
                        self.stdout.write(self.style.ERROR(f"    - {error}"))
                logger.info(f"Stock news updated: {news_result}")
            except Exception as e:
                error_msg = f"Error updating stock news: {str(e)}"
                results["tasks"]["stock_news"] = {"error": error_msg}
                self.stdout.write(self.style.ERROR(f"  ✗ {error_msg}"))
                logger.error(error_msg, exc_info=True)
        else:
            self.stdout.write(self.style.WARNING("\n[2/7] Skipping Stock News Update"))
            results["tasks"]["stock_news"] = {"skipped": True}

        # 3. Run EFS Evolution
        if not options.get('skip_evolution'):
            self.stdout.write(self.style.WARNING("\n[3/7] Running EFS Evolution..."))
            try:
                evolution_result = run_efs_evolution(
                    generation_size=6,
                    top_k=3,
                    mutate_count=3,
                    top_m=10,
                    min_samples=20,
                    window_months=12,
                    eval_frequency="biweekly",
                    step_days=10,
                )
                results["tasks"]["efs_evolution"] = evolution_result
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Cleared {evolution_result['cleared_evaluations']} evaluations"
                    )
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Created: {evolution_result.get('created', 0)}, "
                        f"Evaluated: {evolution_result.get('evaluated', 0)}, "
                        f"Promoted: {evolution_result.get('promoted', 0)}"
                    )
                )
                logger.info(f"EFS evolution completed: {evolution_result}")
            except Exception as e:
                error_msg = f"Error running EFS evolution: {str(e)}"
                results["tasks"]["efs_evolution"] = {"error": error_msg}
                self.stdout.write(self.style.ERROR(f"  ✗ {error_msg}"))
                logger.error(error_msg, exc_info=True)
        else:
            self.stdout.write(self.style.WARNING("\n[3/7] Skipping EFS Evolution"))
            results["tasks"]["efs_evolution"] = {"skipped": True}

        # 4. Update Recommended Questions
        if not options.get('skip_questions'):
            self.stdout.write(self.style.WARNING("\n[4/7] Updating Recommended Questions..."))
            try:
                questions_result = update_recommended_questions(limit=10)
                results["tasks"]["recommended_questions"] = questions_result
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Deleted: {questions_result.get('deleted', 0)} old questions"
                    )
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Created: {questions_result.get('created', 0)} new questions"
                    )
                )
                if questions_result.get('questions'):
                    for q in questions_result['questions'][:3]:  # 显示前3个问题
                        self.stdout.write(self.style.SUCCESS(f"    - {q}"))
                logger.info(f"Recommended questions updated: {questions_result}")
            except Exception as e:
                error_msg = f"Error updating recommended questions: {str(e)}"
                results["tasks"]["recommended_questions"] = {"error": error_msg}
                self.stdout.write(self.style.ERROR(f"  ✗ {error_msg}"))
                logger.error(error_msg, exc_info=True)
        else:
            self.stdout.write(self.style.WARNING("\n[4/7] Skipping Recommended Questions Update"))
            results["tasks"]["recommended_questions"] = {"skipped": True}

        # 5. Update Market Context
        if not options.get('skip_market_context'):
            self.stdout.write(self.style.WARNING("\n[5/7] Updating Market Context..."))
            try:
                market_context_result = update_market_context()
                results["tasks"]["market_context"] = market_context_result
                mc = market_context_result.get("market_context", {})
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Regime: {mc.get('market_regime', 'N/A')}, "
                        f"Cycle: {mc.get('market_cycle', 'N/A')}, "
                        f"Score: {mc.get('market_score', 'N/A')}, "
                        f"Bias: {mc.get('market_bias', 'N/A')}"
                    )
                )
                logger.info(f"Market context updated: {market_context_result}")
            except Exception as e:
                error_msg = f"Error updating market context: {str(e)}"
                results["tasks"]["market_context"] = {"error": error_msg}
                self.stdout.write(self.style.ERROR(f"  ✗ {error_msg}"))
                logger.error(error_msg, exc_info=True)
        else:
            self.stdout.write(self.style.WARNING("\n[5/7] Skipping Market Context Update"))
            results["tasks"]["market_context"] = {"skipped": True}

        # 6. Run Stock Analysis
        if not options.get('skip_analysis'):
            self.stdout.write(self.style.WARNING("\n[6/7] Running Stock Analysis..."))
            try:
                analysis_result = run_stock_analysis()
                results["tasks"]["stock_analysis"] = {
                    "total_stocks": analysis_result["total_stocks"],
                    "success": analysis_result["success"],
                    "failed": analysis_result["failed"],
                    "error_count": len(analysis_result["errors"])
                }
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Success: {analysis_result['success']}/{analysis_result['total_stocks']} stocks"
                    )
                )
                if analysis_result['failed'] > 0:
                    self.stdout.write(
                        self.style.WARNING(f"  ✗ Failed: {analysis_result['failed']} stocks")
                    )
                    for error in analysis_result['errors'][:5]:  # 只显示前5个错误
                        self.stdout.write(self.style.ERROR(f"    - {error.get('symbol')}: {error.get('error')}"))
                logger.info(f"Stock analysis completed: {analysis_result['success']}/{analysis_result['total_stocks']}")
            except Exception as e:
                error_msg = f"Error running stock analysis: {str(e)}"
                results["tasks"]["stock_analysis"] = {"error": error_msg}
                self.stdout.write(self.style.ERROR(f"  ✗ {error_msg}"))
                logger.error(error_msg, exc_info=True)
        else:
            self.stdout.write(self.style.WARNING("\n[6/7] Skipping Stock Analysis"))
            results["tasks"]["stock_analysis"] = {"skipped": True}

        # 7. Update Rankings
        if not options.get('skip_rankings'):
            self.stdout.write(self.style.WARNING("\n[7/7] Updating Rankings..."))
            try:
                ranking_result = update_rankings()
                results["tasks"]["rankings"] = ranking_result
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Factors: {ranking_result.get('factors', {}).get('updated', 0)} updated"
                    )
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Stocks: {ranking_result.get('stocks', {}).get('updated', 0)} updated"
                    )
                )
                logger.info(f"Rankings updated: {ranking_result}")
            except Exception as e:
                error_msg = f"Error updating rankings: {str(e)}"
                results["tasks"]["rankings"] = {"error": error_msg}
                self.stdout.write(self.style.ERROR(f"  ✗ {error_msg}"))
                logger.error(error_msg, exc_info=True)
        else:
            self.stdout.write(self.style.WARNING("\n[7/7] Skipping Rankings Update"))
            results["tasks"]["rankings"] = {"skipped": True}

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = duration

        self.stdout.write(self.style.SUCCESS("\n" + "=" * 80))
        self.stdout.write(self.style.SUCCESS(f"Total Duration: {duration/60:.2f} min"))
        self.stdout.write(self.style.SUCCESS("=" * 80))
        
        logger.info(f"Daily run completed in {duration:.2f} seconds")
