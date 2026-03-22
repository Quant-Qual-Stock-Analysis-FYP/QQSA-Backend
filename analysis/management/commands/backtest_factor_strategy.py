import json
from datetime import date
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from analysis.services.backtest import backtest_dynamic_factor_strategy


class Command(BaseCommand):
    help = "Run dynamic factor-only walk-forward backtest."

    def add_arguments(self, parser):
        parser.add_argument("--start-date", type=str, required=True, help="Backtest start date, e.g. 2025-01-01")
        parser.add_argument("--end-date", type=str, required=True, help="Backtest end date, e.g. 2025-12-31")
        parser.add_argument("--top-n", type=int, default=5, help="Number of stocks to hold each rebalance")
        parser.add_argument("--rebalance-every", type=int, default=20, help="Rebalance every N benchmark trading days")
        parser.add_argument("--factor-lookback-days", type=int, default=120, help="Historical trading days used to choose factor")
        parser.add_argument("--future-days", type=int, default=20, help="Forward trading days used to evaluate factor performance")
        parser.add_argument("--candidate-factor-top-k", type=int, default=10, help="Only test the top K candidate factors by prior score")
        parser.add_argument("--eval-step-days", type=int, default=10, help="Spacing between historical evaluation dates")
        parser.add_argument("--top-m", type=int, default=10, help="Use average forward return of top M ranked stocks in factor evaluation")
        parser.add_argument("--min-samples", type=int, default=20, help="Minimum cross-sectional samples required for each evaluation date")
        parser.add_argument("--initial-capital", type=float, default=1000000.0, help="Initial capital in USD")
        parser.add_argument(
            "--summary-only",
            action="store_true",
            help="Print only the config and summary instead of the full log payload",
        )
        parser.add_argument(
            "--report-only",
            action="store_true",
            help="Print only summary, rebalance_log, and config for report-friendly output",
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Optional path to export the JSON payload to a file",
        )

    def handle(self, *args, **options):
        if options["summary_only"] and options["report_only"]:
            raise CommandError("--summary-only and --report-only cannot be used together")

        try:
            start_date = date.fromisoformat(options["start_date"])
            end_date = date.fromisoformat(options["end_date"])
        except ValueError as exc:
            raise CommandError("Dates must be in ISO format YYYY-MM-DD") from exc

        try:
            result = backtest_dynamic_factor_strategy(
                start_date=start_date,
                end_date=end_date,
                top_n=options["top_n"],
                rebalance_every=options["rebalance_every"],
                factor_lookback_days=options["factor_lookback_days"],
                future_days=options["future_days"],
                candidate_factor_top_k=options["candidate_factor_top_k"],
                eval_step_days=options["eval_step_days"],
                top_m=options["top_m"],
                min_samples=options["min_samples"],
                initial_capital=options["initial_capital"],
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        if options["summary_only"]:
            payload = {
                "config": result.get("config", {}),
                "summary": result.get("summary", {}),
                "rebalances": len(result.get("rebalance_log", [])),
            }
        elif options["report_only"]:
            payload = {
                "summary": result.get("summary", {}),
                "rebalance_log": result.get("rebalance_log", []),
                "config": result.get("config", {}),
            }
        else:
            payload = result

        json_text = json.dumps(payload, indent=2)

        output_path = options.get("output")
        if output_path:
            destination = Path(output_path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json_text + "\n", encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"JSON exported to {destination}"))
        else:
            self.stdout.write(json_text)
