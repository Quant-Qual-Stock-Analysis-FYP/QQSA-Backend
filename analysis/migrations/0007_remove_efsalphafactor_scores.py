from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("analysis", "0006_ragdocumentfundamental"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="efsalphafactor",
            name="last_ic",
        ),
        migrations.RemoveField(
            model_name="efsalphafactor",
            name="last_top_return",
        ),
    ]
