"""
Очистка облака .xyz (class x y z) от мусора.

    python scripts/clean_noise.py examples/datasets/terrain_points.xyz
    python scripts/clean_noise.py cloud.xyz --out exports/cloud_clean.xyz --noise-out exports/cloud_noise.xyz
    python scripts/clean_noise.py cloud.xyz --eval-noise-classes 7 18   # если шум размечен

Пишет два файла:
  *_clean.xyz — облако без мусора (классы сохраняются);
  *_noise.xyz — удалённые точки с кодом 7 (низкий/общий шум) или 18 (высокий шум) —
                откройте его поверх исходного облака во вьюере и проверьте глазами.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data.xyz_loader import XYZLoader          # noqa: E402
from lib.data.xyz_saver import save_points_to_xyz   # noqa: E402
from lib.optimization.filtering import clean, metrics, ALS_PRESET  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out")
    ap.add_argument("--noise-out")
    ap.add_argument("--steps", help="JSON со списком шагов вместо ALS_PRESET")
    ap.add_argument("--eval-noise-classes", type=int, nargs="+",
                    help="классы, которые считать размеченным мусором для метрик")
    a = ap.parse_args()

    # float64: на UTM-координатах float32 даёт шаг 0.5 м
    classes, points = XYZLoader(a.input).load(dtype=np.float64)
    steps = json.loads(Path(a.steps).read_text()) if a.steps else ALS_PRESET
    print(f"Точек: {len(points)}. Шаги:")
    res = clean(points, classes, steps)

    stem = Path(a.input).with_suffix("")
    out = a.out or f"exports/{stem.name}_clean.xyz"
    nout = a.noise_out or f"exports/{stem.name}_noise.xyz"
    save_points_to_xyz(points[res.keep], out, classes=classes[res.keep])
    save_points_to_xyz(points[~res.keep], nout, classes=res.codes[~res.keep])
    print(f"Удалено {(~res.keep).sum()} из {len(points)} "
          f"(низкий шум: {(res.codes == 7).sum()}, высокий: {(res.codes == 18).sum()})")

    if a.eval_noise_classes:
        m = metrics(res.keep, np.isin(classes, a.eval_noise_classes))
        print("Метрики:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    main()
