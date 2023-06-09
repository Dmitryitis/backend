# Generated by Django 4.0.8 on 2023-04-21 16:54

import api.models.project.project_model
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_project_is_deleted'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProjectModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True, verbose_name='Creation time')),
                ('updated_at', models.DateTimeField(auto_now=True, null=True, verbose_name='Update time')),
                ('column_predict', models.CharField(blank=True, max_length=128, null=True)),
                ('type_study', models.CharField(choices=[('fast', 'Fast'), ('slow', 'Slow')], max_length=64)),
                ('save_model', models.FileField(blank=True, null=True, upload_to=api.models.project.project_model.get_project_file_model_path)),
                ('project', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, to='api.project')),
            ],
            options={
                'verbose_name_plural': 'ProjectModelData',
            },
        ),
    ]
