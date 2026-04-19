# Расширения к beeFormer

Этот репозиторий основан на [beeFormer](https://github.com/recombee/beeformer) (обучение языковых эмбеддингов предметов по матрице взаимодействий). Здесь добавлены:

- **Асимметричный beeFormer** — раздельные Q и V, сходство по \(Q V^\top\); тип задаётся флагом `--asym_type` (нужны `--use_asym_model true`).
- **L3AE (закрытая форма)** — по замороженному текстовому энкодеру считаются эмбеддинги предметов, затем в явном виде **S** и **B**, инференс через `SparseKerasEASE`. **Симметричный** и **асимметричный** L3AE опираются на эмбеддинги **симметричного** и **асимметричного** текстового энкодера (SentenceTransformer с одним или с раздельными Q/V выходами) соответственно.

### Типы асимметрии (`--asym_type`)

| Значение | Идея |
|----------|------|
| **`mlp`** | После стандартного pooling два независимых MLP «на голову» (Q и V) с остаточным подходом к исходному эмбеддингу предложения. |
| **`qformer`** | Вместо pooling — несколько слоёв cross-attention: обучаемые запросы читают выход трансформера по токенам и дают два вектора (Q и V). |
| **`prepend`** | Перед трансформером добавляются обучаемые префикс-токены отдельно для Q- и V-ветки (удваивается батч); после блока префиксы отрезаются до pooling, затем `SplitQV` разделяет Q/V. |
| **`dual`** | Две полные копии SentenceTransformer (общий старт весов, дальше независимое дообучение); на выходе — эмбеддинги двух энкодеров как Q и V. |

## Steps to start training the models

1. Create a virtual environment: `python3.10 -m venv beef` and activate it: `source beef/bin/activate`
2. Clone this repository and navigate to the project root: `cd beeformer`
3. Install packages: `pip install -r requirements.txt`
4. Download the data you need: in `_datasets/ml20m`, `_datasets/goodbooks`, or `_datasets/amazbooks` run `source download_data`; for Amazon datasets also run `python preprocess.py`. Other dataset keys in `config.py` (e.g. `amazon-grocery`, `amazon-beauty`) point at paths under `_datasets/`—prepare those files analogously.
5. From the project root run `train.py`, for example:

```bash
python train.py --seed 42 --scheduler None --lr 1e-5 --epochs 5 \
  --dataset goodbooks --sbert sentence-transformers/all-mpnet-base-v2 \
  --max_seq_length 384 --batch_size 1024 --max_output 10000 \
  --sbert_batch_size 200 --use_cold_start true --save_every_epoch true \
  --use_asym_model true --asym_type mlp --model_name my_asym_model
```

## Запуск обучения

См. `requirements.txt`. Ниже — другие режимы (L3AE, ELSA) и оценка.

Закрытая форма L3AE (после обучения или с готовым чекпоинтом SBERT):

```bash
python train.py --use_l3ae_model true --use_asym_model true --asym_type mlp \
  --dataset goodbooks --sbert /path/to/checkpoint ... --model_name out_l3ae
```

Прямой ELSA / AsymELSA без текста: `--use_elsa_model true` или `--use_asym_elsa_model true`.

Оценка: `evaluate_itemsplit.py`, `evaluate_timesplit.py` (аргументы см. в файлах).
