# フォトモザイク画像作成ツール

#  Need
```
pip install numpy Pillow scikit-learn scikit-image
```

# 実行
## Basic
```
python mos.py --target target_image.jpg --tiles ./my_images/
```

## Advance
```
python mos.py --target target_image.py --tiles ./my_images/ --tiles-size 24 --top-k 8
```

## Advance
```
python mos.py --target target_image.py --tiles ./my_images/ --blend 0.2 --output-scale 2
```
