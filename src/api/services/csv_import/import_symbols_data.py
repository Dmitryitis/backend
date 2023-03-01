import csv
import io

from django.conf import settings

from api.models import YahooSymbol
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
