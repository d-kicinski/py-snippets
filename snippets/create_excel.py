from pathlib import Path

import pandas as pd

EXPORT_XLSX = "resources/export.xlsx"
RETAILER_XLSX = "resources/retailer.xlsx"
OUTPUT_XLSX = "out/final.xlsx"
Path(OUTPUT_XLSX).parent.mkdir(exist_ok=True)


def normalize(frame: pd.DataFrame, main_col: str = "EAN", second_col: str = "IDK") -> pd.DataFrame:
    return frame.sort_values(by=[main_col, second_col], ascending=False).drop_duplicates(
        subset=main_col, keep="first"
    )


def create_url(idk: str, language: str = "de") -> str:
    return f"amazon.{language}.com/dp/{idk}/"


def write_excel():
    retailer: pd.DataFrame = pd.read_excel(pd.ExcelFile(RETAILER_XLSX))
    retailer_norm = normalize(retailer, "EAN", "IDK")[["EAN", "IDK"]]

    synthron: pd.DataFrame = pd.read_excel(pd.ExcelFile(EXPORT_XLSX))
    synthron_norm = normalize(synthron, "EAN", "IDK2")[["EAN", "IDK2"]]

    # Add create new column URL based on IDK2 column
    final = pd.merge(retailer_norm, synthron_norm, on="EAN")
    url_column = final.apply(lambda row: create_url(row.IDK2), axis=1)
    final = final.assign(url=url_column.values)

    final.to_excel(OUTPUT_XLSX)


if __name__ == "__main__":
    write_excel()
