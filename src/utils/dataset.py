def dataset_process(ticket, project_pk):
    from api.services.csv_import.import_symbols_data import download_data

    download_data.delay(ticket, project_pk)