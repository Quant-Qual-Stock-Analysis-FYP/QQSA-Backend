# EFS: factor status active/archived only; migrate candidate -> archived

from django.db import migrations


def migrate_candidate_to_archived(apps, schema_editor):
    EfsAlphaFactor = apps.get_model("analysis", "EfsAlphaFactor")
    EfsAlphaFactor.objects.filter(status="candidate").update(status="archived")


class Migration(migrations.Migration):

    dependencies = [
        ("analysis", "0015_add_filename_to_rag_documents"),
    ]

    operations = [
        migrations.RunPython(migrate_candidate_to_archived, migrations.RunPython.noop),
    ]
