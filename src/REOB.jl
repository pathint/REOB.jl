"""
    REOB

Rank Expression Order Based (REOB) 算法包。

本包提供 REO 稳定基因对筛选、REO 分类模型训练/预测/评估，以及
TSP、k-TSP、AUC-TSP 等传统秩序基因对分类基线方法。
"""
module REOB

using Random, Statistics, StatsBase, Distributions, QuadGK
using Lasso, GLM, Combinatorics
using DecisionTree, HypothesisTests

# 导出核心结构和函数
export REOMethod, LassoMethod, RFMethod, VotingMethod
export REOConfig, REOModel
export fit_reo, predict_reo, evaluate_reo, run_permutation_test, generate_test_data

# Tradtional TSP
export TSPModel, KTSPModel, AUCTSPModel 
export fit_tsp, predict_tsp, fit_ktsp, predict_ktsp, fit_auctsp, predict_auctsp 
export evaluate_tsp, evaluate_ktsp, evaluate_auctsp

# 包含核心组件
include("types.jl")       # 定义结构体
include("filters.jl")     # 基因与基因对筛选逻辑
include("training.jl")    # 稳定性选择与模型拟合
include("vote.jl")
include("evaluation.jl")  # 验证与性能指标
include("utils.jl")       # 其他代码
include("statistics.jl")
include("tsp.jl")
include("ktsp.jl")
include("auctsp.jl")

end
