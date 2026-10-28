import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------- Парсинг ----------

def parse_obl_file(path):
    """Возвращает (meta_dict, ndarray shape=(N,4) [rho, T, P, e])."""
    meta = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                body = s.lstrip('#').strip()
                if ':' in body:
                    k, v = body.split(':', 1)
                    meta[k.strip()] = v.strip()
            else:
                break

    arr = np.loadtxt(path, comments='#')
    if arr.ndim == 1:
        arr = arr[None, :]
    return meta, arr


def parse_pair_range(s):
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    return float(nums[0]), float(nums[1])


def parse_grid_type(s):
    out = {}
    for part in s.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            out[k.strip()] = v.strip().lower()
    return out


def load_obl_set(folder='.', pattern='Obl*.txt'):
    files = sorted(Path(folder).glob(pattern))
    records = []
    for p in files:
        meta, arr = parse_obl_file(p)

        name = p.stem
        if 'Region' in meta:
            name = meta['Region'].split(':')[0].strip()

        rho_log = T_log = False
        if 'Grid type' in meta:
            gt = parse_grid_type(meta['Grid type'])
            rho_log = gt.get('rho', '').startswith('log')
            T_log   = gt.get('T',   '').startswith('log')

        rho_range = parse_pair_range(meta['rho range']) if 'rho range' in meta else None
        T_range   = parse_pair_range(meta['T range'])   if 'T range'   in meta else None

        records.append(dict(
            name=name, path=p, meta=meta,
            rho=arr[:, 0], T=arr[:, 1], P=arr[:, 2], e=arr[:, 3],
            rho_log=rho_log, T_log=T_log,
            rho_range=rho_range, T_range=T_range,
            N=len(arr),
        ))
    return records


records = load_obl_set('.')
print(f"Загружено файлов: {len(records)}")

summary = pd.DataFrame([{
    'file':    r['path'].name,
    'name':    r['name'],
    'N':       r['N'],
    'rho_min': r['rho'].min(), 'rho_max': r['rho'].max(),
    'T_min':   r['T'].min(),   'T_max':   r['T'].max(),
    'rho_grid': 'log' if r['rho_log'] else 'lin',
    'T_grid':   'log' if r['T_log']   else 'lin',
} for r in records])
summary

# ---------- Сохранение общего датасета ----------

def save_combined_dataset(records, path='dataset.txt', path_4col=None,
                          fmt='%.10e'):
    """
    Пишет общий датасет:
      • path       — с колонкой region (удобно для группировок в pandas);
      • path_4col  — только rho, T, P, e без служебных колонок.
    Возвращает DataFrame, который был сохранён.
    """
    df = pd.concat([
        pd.DataFrame({
            'region': r['name'],
            'rho':    r['rho'],
            'T':      r['T'],
            'P':      r['P'],
            'e':      r['e'],
        }) for r in records
    ], ignore_index=True)

    header = (
        "# Combined dataset Obl01..Obl20\n"
        f"# N files:   {len(records)}\n"
        f"# N points:  {len(df)}\n"
        "# Columns:   region, rho, T, P, e\n"
        "# Units:     rho [g/cm^3], T [K], P [GPa], e [kJ/g]\n"
    )
    with open(path, 'w') as f:
        f.write(header)
        f.write("region rho T P e\n")
    df.to_csv(path, sep=' ', header=False, index=False,
              mode='a', float_format=fmt)

    if path_4col is not None:
        header4 = (
            "# Combined dataset (4 columns, no region)\n"
            f"# N points:  {len(df)}\n"
            "# Columns:   rho, T, P, e\n"
            "# Units:     rho [g/cm^3], T [K], P [GPa], e [kJ/g]\n"
        )
        with open(path_4col, 'w') as f:
            f.write(header4)
            f.write("rho T P e\n")
        df[['rho', 'T', 'P', 'e']].to_csv(
            path_4col, sep=' ', header=False, index=False,
            mode='a', float_format=fmt)

    print(f"✓ {path}: {len(df):,} строк, {len(df.columns)} колонок")
    if path_4col is not None:
        print(f"✓ {path_4col}: {len(df):,} строк, 4 колонки")
    print(f"  разбивка по регионам: "
          f"{df['region'].nunique()} уникальных region")
    return df


df_all = save_combined_dataset(
    records,
    path='dataset.txt',
    path_4col='dataset_4cols.txt',
)

# sanity-check: чтение обратно
_check = pd.read_csv('dataset.txt', sep=' ', comment='#')
print(f"\nОбратное чтение dataset.txt: shape = {_check.shape}")
_check.head()
