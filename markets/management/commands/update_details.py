from django.core.management.base import BaseCommand
from markets.services.update import update_stock_details


class Command(BaseCommand):
    help = '使用 yahooquery 獲取真實股票基本面數據'

    def handle(self, *args, **kwargs):
        result = update_stock_details()
        
        self.stdout.write(
            self.style.SUCCESS(
                f"成功更新 {result['success']}/{result['total']} 個股票"
            )
        )
        
        if result['failed'] > 0:
            self.stdout.write(
                self.style.WARNING(f"失敗: {result['failed']} 個股票")
            )
            for error in result['errors']:
                self.stdout.write(self.style.ERROR(f"  - {error}"))