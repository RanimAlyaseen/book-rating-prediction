# Private dataset verification summary

This summary provides reproducibility evidence without redistributing the university workbook.

## Fingerprint and profile

- Source filename: `book recomendation final_dataset.xlsx`
- SHA-256: `74AC6A8AF44CBE04D3824B39A14B40634BA859A275819C74D00B8EE4EA40777E`
- Raw dimensions: 350 rows and 10 columns
- Columns: `book_id`, `name`, `author`, `rating`, `year`, `genre`, `price`, `num_reviews`, `num_pages`, `sub_genre`
- Missing values: none
- Exact duplicate rows: none
- Duplicate book identifiers: none
- Removed rows: two records with price equal to zero
- Clean observations: 348
- Positive target: 241 records
- Negative target: 107 records
- Positive prevalence: 69.25%
- Repeated normalized-title groups: one group containing two editions

Review count is strongly right-skewed (sample skewness 7.85). Numeric input correlations with the binary target are weak: year 0.126, reviews 0.109, price 0.012, and page count -0.015. These descriptive values do not establish causality.

## Source comparison

The reports identify Kaggle as the original source. The likely base is *Amazon Top 50 Bestselling Books 2009–2019*, whose Kaggle data card labels the dataset CC0 and contains 550 records with seven fields.

A downloaded copy of that base was compared with the private workbook using title, author, and year:

- 335 of 350 project rows matched a base record on those keys.
- All 335 matched rows retained the same rating.
- All 335 matched rows changed the review count.
- 327 matched rows changed price.
- 39 matched rows changed genre.
- The project adds `book_id`, `num_pages`, and `sub_genre`.

The source and license of those additions and modifications are undocumented. The university workbook is therefore excluded even though the likely base dataset is CC0.

Base dataset: https://www.kaggle.com/datasets/sootersaalu/amazon-top-50-bestselling-books-2009-2019
