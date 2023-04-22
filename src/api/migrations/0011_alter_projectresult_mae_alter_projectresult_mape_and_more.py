# Generated by Django 4.0.8 on 2023-04-22 10:58

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0010_projectresult'),
    ]

    operations = [
        migrations.AlterField(
            model_name='projectresult',
            name='mae',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='projectresult',
            name='mape',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='projectresult',
            name='predictions',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), blank=True, null=True, size=None),
        ),
        migrations.AlterField(
            model_name='projectresult',
            name='r2',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='projectresult',
            name='rmse',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
