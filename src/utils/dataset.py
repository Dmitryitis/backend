def dataset_process(ticket, project_pk):
    from api.services.csv_import.import_symbols_data import download_data

    download_data(ticket, project_pk)