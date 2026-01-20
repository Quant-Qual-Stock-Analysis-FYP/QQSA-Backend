from django.core.management.base import BaseCommand
from markets.services.update import update_stock_news


class Command(BaseCommand):
    help = 'Fetch news with strict filtering and special handling for Yahoo icons'

    def handle(self, *args, **kwargs):
        result = update_stock_news(days_back=3)
        
        self.stdout.write(
            self.style.SUCCESS(
                f"成功更新 {result['success']}/{result['total_stocks']} 個股票，"
                f"共 {result['total_news']} 則新聞"
            )
        )
        
        if result['failed'] > 0:
            self.stdout.write(
                self.style.WARNING(f"失敗: {result['failed']} 個股票")
            )
            for error in result['errors']:
                self.stdout.write(self.style.ERROR(f"  - {error}"))