import csv
import io
from io import StringIO
import yfinance as yf

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction

from api.models import YahooSymbol, ProjectFileData, Project
from vkr.celery import app


@app.task(queue=settings.QUEUE_DEFAULT)
def import_symbols_data(table):
    decoded_file = table.read().decode('utf-8')
    io_string = io.StringIO(decoded_file)

    reader = csv.DictReader(io_string)

    symbols_to_create = []

    for row in reader:
        symbol = row['Symbol']
        name = row["Name"]

        if not YahooSymbol.objects.filter(symbol=symbol, name=name).exists():
            symbols_to_create.append(YahooSymbol(symbol=symbol, name=name))

    YahooSymbol.objects.bulk_create(symbols_to_create)


@app.task(queue=settings.QUEUE_DEFAULT)
def download_data(ticket, project_pk):
    with transaction.atomic():
        data = yf.download(ticket)

        columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(columns)

        for idx, row in data.iterrows():
            data_row = [idx, row['Open'], row['High'], row['Low'], row['Close'], row["Adj Close"], row['Volume']]
            csv_writer.writerow(data_row)

        csv_file = ContentFile(csv_buffer.getvalue().encode('utf-8'), f"{ticket}_{project_pk}.csv")

        project_file_data = ProjectFileData.objects.create(name=f"{ticket}_{project_pk}", size="100", file=csv_file)

        project_instance = Project.objects.filter(id=project_pk).first()

        project_instance.project_file_data = project_file_data
        project_instance.save()
