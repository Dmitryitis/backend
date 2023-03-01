from django import forms
from django.contrib.admin import ModelAdmin

from api.services.csv_import.import_symbols_data import import_symbols_data


class CsvSymbolsImportForm(forms.ModelForm):
    csv_import_file = forms.FileField(required=True)

    def save(self, commit=False):
        csv_import_file = self.cleaned_data.pop("csv_import_file", None)

        if csv_import_file is not None:
            import_symbols_data(
                table=csv_import_file,
            )

        return super(CsvSymbolsImportForm, self).save(commit=False)


class CsvSymbolsImportAdmin(ModelAdmin):
    form = CsvSymbolsImportForm
