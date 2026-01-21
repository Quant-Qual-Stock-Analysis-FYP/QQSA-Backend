"""
Update Market Context Management Command

運行方式：
    python manage.py update_market_context
"""

import logging
from datetime import datetime
from django.core.management.base import BaseCommand

from analysis.services.daily_tasks import update_market_context

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Update market context data using intermarket analysis"

    def handle(self, *args, **options):
        start_time = datetime.now()
        self.stdout.write(self.style.SUCCESS("=" * 80))
        self.stdout.write(self.style.SUCCESS(f"Update Market Context Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"))
        self.stdout.write(self.style.SUCCESS("=" * 80))
        
        try:
            self.stdout.write(self.style.WARNING("\n[1/1] Updating Market Context..."))
            
            result = update_market_context()
            market_context = result.get("market_context", {})
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"  ✓ Cycle: {market_context.get('market_cycle', 'N/A')}, "
                    f"Score: {market_context.get('market_score', 'N/A')}, "
                    f"Bias: {market_context.get('market_bias', 'N/A')}"
                )
            )
            
            # 顯示市場原因（最多顯示前3個）
            reasons = market_context.get("market_reason", [])
            if reasons:
                self.stdout.write(self.style.SUCCESS("\n  Market Reasons:"))
                for i, reason in enumerate(reasons[:3], 1):
                    # 截斷過長的原因
                    display_reason = reason[:150] + "..." if len(reason) > 150 else reason
                    self.stdout.write(self.style.SUCCESS(f"    {i}. {display_reason}"))
            
            logger.info(f"Market context updated successfully: {market_context}")
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.stdout.write(self.style.SUCCESS("\n" + "=" * 80))
            self.stdout.write(self.style.SUCCESS(f"Duration: {duration:.2f} seconds"))
            self.stdout.write(self.style.SUCCESS("=" * 80))
            
            logger.info(f"Update market context completed in {duration:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Error updating market context: {str(e)}"
            self.stdout.write(self.style.ERROR(f"\n  ✗ {error_msg}"))
            logger.error(error_msg, exc_info=True)
            raise
