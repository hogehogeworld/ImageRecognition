## 前処理なしの結果

```python
KNN
	Elapsed time is 33.443956 seconds.
	accuracy=63.30
ユークリッド距離
	Elapsed time is 0.052709 seconds.
	accuracy=80.20
正規化相関
	Elapsed time is 0.050822 seconds.
	accuracy=80.20
主成分分析
	Elapsed time is 4.763235 seconds.
	accuracy=94.90
部分空間法
	Elapsed time is 0.385812 seconds.
	accuracy=94.90
マハラノビス距離
	Elapsed time is 2.457265 seconds.
	accuracy=82.40
```

## 前処理ありの結果

フィッシャーの多重判別分析で次元を落としたのちに、分類した結果


```python
ユークリッド距離
	Elapsed time is 0.301467 seconds.
	accuracy=82.40
正規化相関
	Elapsed time is 0.286572 seconds.
	accuracy=82.40
部分空間法
	Elapsed time is 0.903892 seconds.
	accuracy=80.80
```
