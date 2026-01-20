from django.db import migrations, models
import django.db.models.deletion
from django.contrib.postgres.fields import ArrayField


class Migration(migrations.Migration):

    dependencies = [
        ("analysis", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="RagDocumentFundamental",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("content", models.TextField()),
                ("source", models.CharField(blank=True, max_length=255)),
                ("is_manual", models.BooleanField(default=False)),
                ("embedding", ArrayField(base_field=models.FloatField(), default=list, blank=True, size=None)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "stock",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="rag_fundamentals",
                        to="markets.stock",
                    ),
                ),
            ],
        ),
        migrations.AddIndex(
            model_name="ragdocumentfundamental",
            index=models.Index(fields=["stock", "created_at"], name="analysis_rag_stock_fund_idx"),
        ),
    ]
