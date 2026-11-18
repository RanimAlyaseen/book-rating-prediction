# Data access

The original university workbook is intentionally excluded from this repository.

The project report associates the data with Kaggle's *Amazon Top 50 Bestselling Books 2009–2019* dataset, whose Kaggle data card identifies the base dataset as CC0. The university workbook is not an unchanged copy: it adds `book_id`, `num_pages`, and `sub_genre`, changes most review and price values, changes some genres, and contains 15 title/author/year records that do not match the downloaded Kaggle file. The provenance and redistribution terms for those additions are not documented.

To reproduce the recorded experiment, place an authorized copy of the university workbook at:

```text
data/raw/book_rating_dataset.xlsx
```

Then run:

```powershell
python -m book_rating.train --data data/raw/book_rating_dataset.xlsx
```

`data/synthetic_books.csv` contains 24 fabricated records for demonstrations and test development. It must not be used to reproduce the reported model scores. Run `python -m book_rating.demo` to generate a clearly labeled local demonstration model and isolated outputs; do not pass this fixture to the research training command.

Base dataset reference: https://www.kaggle.com/datasets/sootersaalu/amazon-top-50-bestselling-books-2009-2019
