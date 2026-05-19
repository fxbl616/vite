# NBA Data Placeholder

Put MART-style NBA files here:

```text
data/nba/nba_train.npy
data/nba/nba_test.npy
```

The NBA adapter also checks common external locations such as:

```text
external/MART/datasets/nba/
../MART/datasets/nba/
```

You can inspect or copy an existing local dataset with:

```bash
python scripts/prepare_nba_data.py
python scripts/prepare_nba_data.py --source_root <folder-containing-npy> --copy
```

Do not use `output/synthetic_nba_mart` for real experiments. It is only a smoke-test dataset generated during development.
