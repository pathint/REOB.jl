# API 参考

本页由 `Documenter.jl` 从源码 docstring 生成，用于检查公开 API 文档是否
和当前实现保持一致。

## 类型与配置

```@docs
REOB
REOB.REOMethod
REOB.REOConfig
REOB.REOModel
REOB.TSPModel
REOB.KTSPModel
REOB.AUCTSPModel
```

## REOB 主流程

```@docs
REOB.fit_reo
REOB.predict_reo
REOB.evaluate_reo
REOB.run_permutation_test
REOB.generate_test_data
```

## 传统 TSP 方法

```@docs
REOB.fit_tsp
REOB.predict_tsp
REOB.evaluate_tsp
REOB.fit_ktsp
REOB.predict_ktsp
REOB.evaluate_ktsp
REOB.fit_auctsp
REOB.predict_auctsp
REOB.evaluate_auctsp
```
