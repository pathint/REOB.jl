export REOMethod, LassoMethod, RFMethod, VotingMethod, REOConfig, REOModel

"""
    REOMethod

REOB 训练策略枚举。

- `LassoMethod`：使用 Lasso/Elastic Net 路径筛选并加权基因对。
- `RFMethod`：使用随机森林树桩模型筛选稳定基因对。
- `VotingMethod`：使用多数投票规则选择基因对子集。
"""
@enum REOMethod LassoMethod RFMethod VotingMethod

"""
    REOConfig(; kwargs...)

REOB 算法超参数配置。

# 关键字参数

- `method::REOMethod=RFMethod`：训练策略，决定使用随机森林、Lasso 或多数投票分支。
- `low_rank_q::Float64=0.2`：低表达基因秩分位过滤阈值。
- `top_diff_n::Int=5000`：差异秩次预筛选保留基因数。
- `fisher_n_top::Int=5000`：Fisher 基因对预筛选的保留数量，当前主流程默认使用 BQC。
- `max_occurrence::Int=2`：同一基因最多参与的最终候选基因对数量。
- `p_val_cutoff::Float64=0.05`：混淆因子审计的显著性阈值。
- `cor_threshold::Float64=0.90`：相关性剪枝阈值。
- `ss_iterations::Int=1000`：稳定性选择迭代次数。
- `ss_ratio::Float64=0.8`：每轮稳定性选择的分层抽样比例。
- `ss_threshold::Float64=0.7`：稳定性选择保留阈值。
- `forest_trees::Int=100`：随机森林树数配置，当前实现保留该字段但未直接读取。
- `forest_depth::Int=1`：随机森林深度配置，当前实现保留该字段但未直接读取。
- `lasso_lambda::Symbol=:min`：Lasso 路径的目标 lambda 选择方式。
- `bqc_threshold::Float64=3.0`：BQC 基因对稳定性得分阈值。
- `p0_threshold::Float64=0.2`：对照组序关系远离 0.5 的稳定性阈值。
- `target_n::Int=15`：目标特征对数量。
- `verbose::Bool=false`：是否输出训练过程日志。

# 示例

```julia
cfg = REOConfig(method=VotingMethod, target_n=5, ss_iterations=50)
```
"""
Base.@kwdef struct REOConfig
    method::REOMethod = RFMethod        # 选择 LassoMethod 或 RFMethod
    
    # --- 通用筛选参数 ---
    low_rank_q::Float64 = 0.2          # 低表达筛选阈值 
    top_diff_n::Int = 5000              # 差异基因保留数 
    fisher_n_top::Int = 5000           # Fisher 检验对数 
    max_occurrence::Int = 2            # 基因去冗余上限 
	p_val_cutoff::Float64 = 0.05	   # cutoff value for confounders detection
    cor_threshold::Float64 = 0.90      # 相关性剪枝阈值 
    
    # 稳定性选择参数
    ss_iterations::Int = 1000          # 稳定性选择迭代次数
    ss_ratio::Float64 = 0.8            # 下采样比例
    ss_threshold::Float64 = 0.7        # 频率阈值 
    
	# 森林模型参数
    forest_trees::Int = 100            # 树的数量
    forest_depth::Int = 1              # 决策深度(1 为 Stump)

    # --- Lasso 特定参数 (参考 lassoREO.jl 逻辑) ---
    lasso_lambda::Symbol = :min        # :min 或 :1se

	# BQC 相关配置
	bqc_threshold::Float64 = 3.0  # 置信度阈值 (0.5~1.0)
	p0_threshold::Float64 = 0.2   # 置信度阈值 (0.5~1.0)

    target_n::Int = 15                 # 最终特征数 
    verbose::Bool = false
end

"""
    REOModel

训练完成的 REO 模型。

# 字段

- `config::REOConfig`：训练时使用的配置。
- `final_pairs::Vector{Tuple{String,String}}`：方向已对齐的最终基因对，`g1 > g2` 表示特征值为 `true`。
- `weights::Vector{Float64}`：各基因对权重，已归一化到可直接用于预测评分。
- `intercept::Float64`：线性评分截距或投票偏置。
"""
struct REOModel
    config::REOConfig
    final_pairs::Vector{Tuple{String, String}}
    weights::Vector{Float64}
    intercept::Float64
end

"""
    TSPModel

Top Scoring Pair (TSP) 二分类模型。

# 字段

- `gene_i::Int`：第一个基因在训练矩阵中的行号。
- `gene_j::Int`：第二个基因在训练矩阵中的行号。
- `gene_names::Tuple{String,String}`：对应的基因名。
- `score::Float64`：训练时得到的主评分 `|p1 - p0|`。
- `p0::Float64`：对照组中 `gene_i < gene_j` 的比例。
- `p1::Float64`：病例组中 `gene_i < gene_j` 的比例。

预测时若 `p1 > p0`，则 `gene_i < gene_j` 支持正类；否则方向反转。
"""
struct TSPModel
    gene_i::Int
    gene_j::Int
    gene_names::Tuple{String, String}
    score::Float64
    p0::Float64  # P(Xi < Xj | Class 0)
    p1::Float64  # P(Xi < Xj | Class 1)
end

"""
    KTSPModel

k-Top Scoring Pairs (k-TSP) 多基因对多数投票模型。

# 字段

- `pairs::Vector{Tuple{Int,Int}}`：训练时选出的基因索引对。
- `gene_names::Vector{Tuple{String,String}}`：基因名对，顺序与 `pairs` 一致。
- `scores::Vector{Float64}`：每个基因对的主评分。
- `p_directions::Vector{Bool}`：方向标记，`true` 表示 `gene_i < gene_j` 投给正类。
- `k::Int`：最终保留的基因对数量，通常保持为奇数以减少平局。
"""
struct KTSPModel
    pairs::Vector{Tuple{Int, Int}}        # 基因索引对
    gene_names::Vector{Tuple{String, String}}
    scores::Vector{Float64}               # 每对的 Delta 得分
    p_directions::Vector{Bool}             # 每对的方向：true 表示 Xi < Xj 指向 Class 1
    k::Int
end

"""
    AUCTSPModel

基于 AUC 排序准则选择基因对的 TSP 变体模型。

# 字段

- `pairs::Vector{Tuple{Int,Int}}`：训练时选出的基因索引对。
- `gene_names::Vector{Tuple{String,String}}`：基因名对，顺序与 `pairs` 一致。
- `auc_scores::Vector{Float64}`：每个基因对的 AUC 得分。
- `directions::Vector{Int}`：方向标记，`1` 表示 `gene_i < gene_j` 支持正类，`-1` 表示反向。
- `k::Int`：最终保留的基因对数量。
"""
struct AUCTSPModel
    pairs::Vector{Tuple{Int, Int}}
    gene_names::Vector{Tuple{String, String}}
    auc_scores::Vector{Float64}
    directions::Vector{Int} # 1: Xi < Xj -> Class 1; -1: Xi > Xj -> Class 1
    k::Int
end
