# Generated migration for adding ranking fields

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0008_remove_efsfactor_stock_delete_efsweight_and_more'),
    ]

    operations = [
        # Add ranking fields to EfsAlphaFactor
        migrations.AddField(
            model_name='efsalphafactor',
            name='factor_rank',
            field=models.PositiveIntegerField(blank=True, help_text='因子排名（1-based，基于 last_score）', null=True),
        ),
        migrations.AddField(
            model_name='efsalphafactor',
            name='factor_universe_size',
            field=models.PositiveIntegerField(blank=True, help_text='排名时的因子总数', null=True),
        ),
        migrations.AddIndex(
            model_name='efsalphafactor',
            index=models.Index(fields=['factor_rank'], name='analysis_ef_factor__idx'),
        ),
        
        # Add ranking fields to AnalysisResult
        migrations.AddField(
            model_name='analysisresult',
            name='stock_rank',
            field=models.PositiveIntegerField(blank=True, help_text='股票排名（1-based，基于 overall_score）', null=True),
        ),
        migrations.AddField(
            model_name='analysisresult',
            name='stock_universe_size',
            field=models.PositiveIntegerField(blank=True, help_text='排名时的股票总数', null=True),
        ),
        migrations.AddIndex(
            model_name='analysisresult',
            index=models.Index(fields=['stock_rank'], name='analysis_an_stock_r_idx'),
        ),
    ]
