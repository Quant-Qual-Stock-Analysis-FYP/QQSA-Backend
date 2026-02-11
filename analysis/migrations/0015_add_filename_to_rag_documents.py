# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0014_remove_efsfactor_stock_delete_efsweight_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='ragdocument',
            name='filename',
            field=models.CharField(blank=True, help_text='Original filename if uploaded from file', max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='ragdocumentfundamental',
            name='filename',
            field=models.CharField(blank=True, help_text='Original filename if uploaded from file', max_length=255, null=True),
        ),
    ]
