# src/filters.jl

using Statistics, StatsBase, HypothesisTests, Combinatorics


"""
    preprocess_filters(data, labels, gene_ids, cfg[, confounders])

根据 `REOConfig` 执行 REOB 预筛选流程。

该函数依次执行低表达过滤、差异秩次过滤、BQC 基因对筛选、可选混淆因子
审计、hub 基因剪枝和相关性剪枝，返回方向已对齐的候选基因对索引和
样本 × 基因对的二值特征矩阵。

# 参数

- `data`：基因 × 样本表达矩阵。
- `labels`：二分类标签，取值应为 `0` 和 `1`。
- `gene_ids`：基因名称；当前预筛选仅保留该参数以匹配训练入口。
- `cfg`：REOB 配置。
- `confounders`：可选协变量向量列表，用于剔除与协变量显著相关的基因对。

# 返回值

`(pairs, X)`，其中 `pairs` 为基因行号元组，`X` 为样本 × 基因对特征矩阵。
"""
function preprocess_filters(
    data::Matrix{<:Real}, 
    labels::AbstractVector, 
    gene_ids::Vector, 
    cfg::REOConfig, 
    confounders::Union{Nothing, Vector{<:AbstractVector}} = nothing
)
    # 1. 过滤低表达 (利用 cfg.low_rank_q)
    keep_low = filter_low_rank_genes(data, cfg.low_rank_q; verbose=cfg.verbose)

    # 2. 差异秩次过滤 (利用 cfg.top_diff_n)
    selected_genes = filter_diff_rank_genes(data, labels, keep_low;
                                            top_n=cfg.top_diff_n, verbose=cfg.verbose)

    # 3. BQC 生物学稳定性审计，剔除组内序关系不稳定的基因对。
    all_pairs = collect(combinations(selected_genes, 2))
    pairs_initial = filter_pairs_by_bqc(all_pairs, data, labels, keep_low, cfg::REOConfig)
    cfg.verbose && println(">>> 基因对经BQC阈值筛选，余 $(length(pairs_initial)) 对基因。")

    length(pairs_initial) == 0 && error("0对基因剩余。可放宽基因对筛选参数，如bqc_threshold和p0_threshold，重新尝试。")

    # 4. 混淆因子审计 (Confounding Factor Audit, 利用cfg.p_val_cutoff)
    if !isnothing(confounders) && !isempty(confounders)
        cfg.verbose && println(">>> 正在针对 $(length(confounders)) 个混淆因子进行独立性审计...")
        
        keep_mask = fill(true, length(pairs_initial))
        p_val_cutoff = cfg.p_val_cutoff # 从 cfg 中读取阈值

        for (i, pair) in enumerate(pairs_initial)
            g1_idx, g2_idx = pair
            # 获取该对子在所有样本中的二值化序关系
            reo_vec = data[g1_idx, :] .> data[g2_idx, :]
            
            for cf_vec in confounders
                if is_confounded(reo_vec, cf_vec, p_val_cutoff)
                    keep_mask[i] = false
                    break # 只要与任何一个混淆因子相关，立即剔除
                end
            end
        end
        
        n_removed = count(!, keep_mask)
        pairs_initial = pairs_initial[keep_mask]
        cfg.verbose && println("    审计完成：剔除了 $n_removed 个与协变量显著相关的基因对。")
    end

    # 5. 去中心化 (利用 cfg.max_occurrence)
    pairs_pruned = prune_hub_genes(pairs_initial, cfg.max_occurrence, verbose = cfg.verbose)

    # 6. 构建方向与正类对齐的特征矩阵。
    X_initial, pairs_pruned = build_feature_matrix_aligned(data, pairs_pruned, labels)
    
    # 7. 相关性剪枝 (利用 cfg.cor_threshold)
    final_pairs, X_final = drop_correlated_features(X_initial, pairs_pruned, cfg.cor_threshold)

    return final_pairs, X_final
end

"""
    filter_low_rank_genes(data, threshold=0.2; verbose=false)

剔除跨样本中位表达秩分位不高于 `threshold` 的基因。

返回保留基因在 `data` 中的行号。
"""
function filter_low_rank_genes(data::Matrix{<:Real}, threshold=0.2; verbose=false)
    n_genes, n_samples = size(data)
    # 计算每个样本内基因的百分比秩次 (0~1)
    percentile_ranks = Matrix{Float64}(undef, n_genes, n_samples)
    Threads.@threads for j in 1:n_samples
        percentile_ranks[:, j] .= tiedrank(data[:, j]) ./ n_genes
    end
    
    # 筛选逻辑：中位秩次处于底部 threshold 的基因被剔除
    keep_indices = findall(i -> median(percentile_ranks[i, :]) > threshold, 1:n_genes)
    
    verbose && println("低表达筛选：从 $(n_genes) 个基因中保留了 $(length(keep_indices)) 个。")
    return keep_indices
end

"""
    filter_diff_rank_genes(data, labels, gene_indices; top_n=500, verbose=false)

在指定基因集合内计算两类样本的平均秩分位差异，返回差异最大的
`top_n` 个基因行号。
"""
function filter_diff_rank_genes(data::Matrix{<:Real}, labels::AbstractVector, gene_indices::Vector{Int}; top_n=500, verbose=false)
    n_samples = size(data, 2)
    n_genes_subset = length(gene_indices)
    
    # 仅对初步保留的基因计算秩次
    sub_data = data[gene_indices, :]
    percentile_ranks = Matrix{Float64}(undef, n_genes_subset, n_samples)
    for j in 1:n_samples
        percentile_ranks[:, j] .= tiedrank(sub_data[:, j]) ./ n_genes_subset
    end
    
    idx1 = findall(x -> x == 1, labels)
    idx0 = findall(x -> x == 0, labels)
    
    # 计算两组间的均值差异 (Absolute Mean Difference)
    diffs = [abs(mean(percentile_ranks[i, idx1]) - mean(percentile_ranks[i, idx0])) for i in 1:n_genes_subset]
    
    # 选取差异最大的前 top_n 个基因
    p = sortperm(diffs, rev=true)
    selected_internal_indices = p[1:min(top_n, length(p))]
    
    final_indices = gene_indices[selected_internal_indices]
    verbose && println("差异筛选：保留了前 $(length(final_indices)) 个高差异基因。")
    return final_indices
end

"""
    get_top_pairs_parallel_fisher(data, labels, gene_indices; n_top=5000, verbose=false)

使用 Fisher 精确检验按显著性筛选候选基因对。

该函数保留为底层筛选工具；REOB 默认训练流程使用 BQC 筛选。
"""
function get_top_pairs_parallel_fisher(data::Matrix{<:Real}, labels::AbstractVector, gene_indices::Vector{Int}; n_top=5000, verbose=false)
    #idx1 = findall(x -> x == 1, labels)
    #idx0 = findall(x -> x == 0, labels)
    idx1 = findall(==(1), labels)
    idx0 = findall(==(0), labels)
    n1, n0 = length(idx1), length(idx0)
    
    # 生成所有可能的基因对组合
    all_pairs = collect(combinations(gene_indices, 2))
    n_pairs = length(all_pairs)
    p_values = Vector{Float64}(undef, n_pairs)
    
    # 并行计算每对基因的 Fisher 检验 P 值
    Threads.@threads for i in 1:n_pairs
        g1, g2 = all_pairs[i]
        # 统计在 Label=1 中 g1 > g2 的数量
        c1 = sum(data[g1, idx1] .> data[g2, idx1])
        # 统计在 Label=0 中 g1 > g2 的数量
        c0 = sum(data[g1, idx0] .> data[g2, idx0])
        # 构建 2x2 混淆矩阵
        #          g1>g2  g1<=g2
        # Label 1:  c1    n1-c1
        # Label 0:  c0    n0-c0
        ft = FisherExactTest(c1, n1-c1, c0, n0-c0)
        p_values[i] = pvalue(ft)
    end
    
    # 按 P 值升序排列
    sp = sortperm(p_values)
    top_indices = sp[1:min(n_top, n_pairs)]
    
    verbose && println("Fisher exact test：获得 $(min(n_top, n_pairs)) 基因对。")
    return all_pairs[top_indices], p_values[top_indices]
end


"""
    filter_pairs_by_bqc(pairs, data, labels, keep_low, cfg)

使用贝叶斯质量控制（BQC）过滤并排序候选基因对。
"""
function filter_pairs_by_bqc(pairs, data, labels, keep_low, cfg::REOConfig)
    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0 = length(idx0)
    n1 = length(idx1)

    # 1. 并行估计全局 Tau
    cfg.verbose && println(">>> 正在并行估计全局 Tau 值...")
    tau_res = estimate_global_tau_parallel(data[keep_low,:]; verbose=cfg.verbose)
    tau = tau_res.mean

    bqc_threshold = cfg.bqc_threshold
    p0_threshold  = cfg.p0_threshold
    threshold_dict = generate_bqc_threshold_dict(n0, n1, tau, bqc_threshold, p0_threshold; verbose=cfg.verbose)
    cfg.verbose && println(">>> 基因对筛选阈值条件，共 $(length(threshold_dict)) 条记录。")
    pairs_initial = filter_pairs_with_dict(pairs, data, labels, threshold_dict; verbose = cfg.verbose)

    n_pairs = length(pairs_initial)
    # 2. 预分配结果数组 (存储 pair, score, p0_diff)
    # 使用 Thread-local 模式或直接并行填充然后过滤
    results = Vector{NamedTuple{(:pair, :score, :p0_diff), Tuple{Tuple{Int, Int}, Float64, Float64}}}(undef, n_pairs)
    
    cfg.verbose && println(">>> 开始并行计算每个基因对的 BQC 分数 (对子总数: $n_pairs)...")

    # 3. 并行计算分数
    Threads.@threads for i in 1:n_pairs
        g1, g2 = pairs_initial[i]
        
        # 快速计算频数 (使用 @views 避免内存拷贝)
        @views k0 = sum(data[g1, idx0] .> data[g2, idx0])
        p0 = k0 / n0
        p0_diff = abs(p0 - 0.5)

        @views k1 = sum(data[g1, idx1] .> data[g2, idx1])
        
        # 调用您实现的增强型 BQC 函数
        score = calculate_enhanced_bqc(k1, n1, p0, tau)
        
        results[i] = (pair = (g1, g2), score = score, p0_diff = p0_diff)
    end


    # 5. 两级排序：
    # 第一关键字：score (倒序)
    # 第二关键字：p0_diff (倒序)
    sort!(results, by = x -> (x.score, x.p0_diff), rev = true)

    # 6. 提取并返回排序后的基因对
    sorted_pairs = [x.pair for x in results]

    return sorted_pairs
end

"""
    filter_pairs_with_dict(pairs, data, labels, threshold_dict; verbose=false)

使用 BQC 临界数字典筛选候选基因对。

`threshold_dict` 的键是对照组中 `g1 > g2` 的次数 `k0`，值是病例组中
达到阈值所需的最小 `k1`。函数返回通过该阈值判断的候选子集。
"""
function filter_pairs_with_dict(pairs, data, labels, threshold_dict::Dict; verbose = false)
    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0 = length(idx0)/2 
    # 预分配结果容器
    keep_mask = fill(false, length(pairs))
    
    Threads.@threads for i in 1:length(pairs)
        g1, g2 = pairs[i]
        
        # 1. 快速计算频数
        k0 = sum(data[g1, idx0] .> data[g2, idx0])
        
        # 2. 查字典：如果 k0 不在字典里，说明 p0 不够稳，直接跳过
        limit = get(threshold_dict, k0, nothing)
        
        if !isnothing(limit)
            k1 = sum(data[g1, idx1] .> data[g2, idx1])
            
            # 3. 极速判断是否落入贝叶斯显著区间
			if (k0 < n0 && k1 >= limit) || (k0 > n0 && k1 <= limit)
                keep_mask[i] = true
            end
        end
    end
    
    verbose && println("    BQC过滤完成：保留 $(sum(keep_mask)) 个稳定翻转基因对。")
    return pairs[keep_mask]
end


"""
    prune_hub_genes(pairs, max_occurrence=2; verbose=false)

限制同一基因在候选基因对中的出现次数，降低 hub 基因造成的过拟合风险。
"""
function prune_hub_genes(pairs, max_occurrence=2; verbose = false)
    gene_counts = Dict{Int, Int}()
    final_pairs = []
    
    for (g1, g2) in pairs
        c1 = get(gene_counts, g1, 0)
        c2 = get(gene_counts, g2, 0)
        
        if c1 < max_occurrence && c2 < max_occurrence
            push!(final_pairs, (g1, g2))
            gene_counts[g1] = c1 + 1
            gene_counts[g2] = c2 + 1
        end
    end
	verbose && println("After pruning: 剩余 $(length(final_pairs)) 基因对。")
    return final_pairs
end

"""
    build_feature_matrix_aligned(data, pairs, labels)

构建方向与正类对齐的二值特征矩阵。

若某个基因对 `g1 > g2` 在正类中出现频率低于负类，则翻转为
`g2 > g1`，使每个特征的 `true` 值尽量指向正类。
"""
function build_feature_matrix_aligned(data::Matrix{<:Real}, pairs, labels::AbstractVector)
    n_samples = size(data, 2)
    n_pairs = length(pairs)
    
    # 预分配结果
    X = BitArray(undef, (n_samples, n_pairs))
    new_pairs = Vector{Tuple{Int, Int}}(undef, n_pairs)

    # 获取类别索引
    pos_idx = findall(==(1), labels)
    neg_idx = findall(==(0), labels)
    n_pos = length(pos_idx)
    n_neg = length(neg_idx)

    Threads.@threads for j in 1:n_pairs
        g1, g2 = pairs[j]
        
        # 计算在两组中 g1 > g2 出现的频率
        @views p_pos = sum(data[g1, pos_idx] .> data[g2, pos_idx]) / n_pos
        @views p_neg = sum(data[g1, neg_idx] .> data[g2, neg_idx]) / n_neg

        # 核心逻辑：如果 g1 > g2 在正类中出现的概率更低，则翻转这对基因
        if p_pos < p_neg
            actual_g1, actual_g2 = g2, g1
            new_pairs[j] = (g2, g1)
        else
            actual_g1, actual_g2 = g1, g2
            new_pairs[j] = (g1, g2)
        end

        # 填充特征矩阵
        @views X[:, j] .= data[actual_g1, :] .> data[actual_g2, :]
    end

    return X, new_pairs
end
"""
    drop_correlated_features(X, pairs, threshold=0.95)

按特征相关系数阈值剔除冗余基因对。
"""
function drop_correlated_features(X::AbstractMatrix, pairs, threshold=0.95)
    n_features = size(X, 2)
    keep = trues(n_features)
    # 计算相关系数矩阵
    cor_mat = cor(X)
    
    for i in 1:n_features
        if !keep[i] continue end
        for j in (i+1):n_features
            if keep[j] && abs(cor_mat[i, j]) > threshold
                keep[j] = false
            end
        end
    end
    
    return pairs[keep], X[:, keep]
end



"""
    is_confounded(reo_vec, cf_vec, p_threshold)

判断一个基因对序关系是否与混淆因子显著相关。

连续协变量使用 Welch t 检验，离散协变量使用卡方独立性检验；无法可靠计算
时返回 `false`，避免因协变量分组过小误删特征。离散协变量支持字符串、
符号等未预先整数编码的分类值。
"""
function is_confounded(reo_vec::AbstractVector{Bool}, cf_vec::AbstractVector, p_threshold::Float64)
    # 1. 如果混淆因子是连续变量 (如Age)
    if eltype(cf_vec) <: AbstractFloat
        # 使用不平衡方差 T 检验 (Welch's T-test)
        # 比较“序关系为0”和“序关系为1”的两组样本在混淆因子上的均值差异
        group0 = cf_vec[reo_vec .== 0]
        group1 = cf_vec[reo_vec .== 1]
        
        # 鲁棒性检查：如果某一组样本太少或完全没有变异，视为无法审计（或保守剔除）
        if length(group0) < 5 || length(group1) < 5 return false end
        if std(group0) ≈ 0 && std(group1) ≈ 0 return false end
        
        return pvalue(UnequalVarianceTTest(group0, group1)) < p_threshold

    # 2. 如果混淆因子是离散/分类型变量 (如 Sex, Batch, Center)
    else
        # 构建 2 x K 列联表 (K 是分类数量)
        # 行：REO (0/1), 列：Confunder Categories
        categories = unique(cf_vec)
        length(categories) > 1 || return false

        cf_to_col = Dict{Any, Int}(category => i for (i, category) in enumerate(categories))
        tbl = zeros(Int, 2, length(categories))
        for (reo, cf) in zip(reo_vec, cf_vec)
            tbl[reo ? 2 : 1, cf_to_col[cf]] += 1
        end
        
        # 使用卡方独立性检验 (ChisqTest)
        # 如果样本量极小，可考虑在此处扩展为 Fisher 精确检验
        try
            return pvalue(ChisqTest(tbl)) < p_threshold
        catch
            return false # 如果由于维度问题计算失败，默认通过
        end
    end
end

"""
    filter_genes(data, labels, gene_ids, cfg)

执行 TSP 系列模型共用的基因级预筛选。

该函数只返回基因行号，不构建基因对或特征矩阵；完整 REOB 预筛选请使用
[`preprocess_filters`](@ref)。
"""
function filter_genes(
    data::Matrix{<:Real}, 
    labels::AbstractVector, 
    gene_ids::Vector, 
    cfg::REOConfig 
)
    # 1. 过滤低表达 (利用 cfg.low_rank_q)
    keep_low = filter_low_rank_genes(data, cfg.low_rank_q; verbose=cfg.verbose)

    # 2. 差异秩次过滤 (利用 cfg.top_diff_n)
    selected_genes = filter_diff_rank_genes(data, labels, keep_low;
                                            top_n=cfg.top_diff_n, verbose=cfg.verbose)
	return selected_genes
end
