# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('corenlp', '__first__'),
    ]

    operations = [
        migrations.CreateModel(
            name='QueryMap',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('query', models.TextField(help_text='Search query entered by user')),
                ('document', models.ForeignKey(to='corenlp.Document', help_text='Result')),
            ],
        ),
    ]
