from django.core.management.base import BaseCommand
from analysis.services.daily_tasks import run_efs_evolution


class Command(BaseCommand):
    help = "Run one EFS evolution cycle (LLM gen -> evaluation -> selection)."

    def add_arguments(self, parser):
        parser.add_argument("--generation", type=int, default=6)
        parser.add_argument("--top-k", type=int, default=3)
        parser.add_argument("--mutate", type=int, default=3)
        parser.add_argument("--top-m", type=int, default=10)
        parser.add_argument("--min-samples", type=int, default=20)
        parser.add_argument("--window-months", type=int, default=12)
        parser.add_argument("--eval-frequency", type=str, default="biweekly")
        parser.add_argument("--step-days", type=int, default=10)

    def handle(self, *args, **options):
        result = run_efs_evolution(
            generation_size=options["generation"],
            top_k=options["top_k"],
            mutate_count=options["mutate"],
            top_m=options["top_m"],
            min_samples=options["min_samples"],
            window_months=options["window_months"],
            eval_frequency=options["eval_frequency"],
            step_days=options["step_days"],
        )
        self.stdout.write(self.style.SUCCESS(f"EFS evolve done: {result}"))
