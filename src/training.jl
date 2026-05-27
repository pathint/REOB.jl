# src/training.jl

using Lasso, DecisionTree, GLM, Random, Statistics, StatsBase

"""
    fit_reo(data, labels, gene_ids, cfg; confounders=nothing)

训练 REOB 分类模型。

# 参数

- `data`：基因 × 样本表达矩阵。
- `labels`：样本标签，二分类取值为 `0` 和 `1`。
- `gene_ids`：与 `data` 行顺序一致的基因名称。
- `cfg`：`REOConfig` 配置对象。
- `confounders`：可选协变量列表；传入后会剔除与协变量显著相关的基因对。

# 返回值

返回 `REOModel`，可传给 [`predict_reo`](@ref) 或 [`evaluate_reo`](@ref)。

# 示例

```julia
cfg = REOConfig(method=VotingMethod, target_n=5)
model = fit_reo(data, labels, genes, cfg)
```
"""
function fit_reo(data::Matrix{<:Real}, labels::AbstractVector, gene_ids::Vector, cfg::REOConfig;
		 confounders::Union{Nothing, Vector{<:AbstractVector}} = nothing)

    # --- 阶段 1: 共享预筛选 ---
    final_pairs_idx, X = preprocess_filters(data, labels, gene_ids, cfg, confounders)

    if isempty(final_pairs_idx)
        error("预筛选后未发现有效基因对，请放宽过滤参数。")
    end

    # --- 阶段 2: 根据方法进行初步训练 ---
    initial_model = if cfg.method == VotingMethod
		if length(final_pairs_idx) > 128
			final_pairs_idx = final_pairs_idx[1:128]
			X = X[:, 1:128]
		end
		_fit_voting_strategy(X, labels, gene_ids, final_pairs_idx, cfg)
	elseif cfg.method == RFMethod
        _fit_rf_strategy(X, labels, gene_ids, final_pairs_idx, cfg)
    elseif cfg.method == LassoMethod
        _fit_lasso_strategy(X, labels, gene_ids, final_pairs_idx, cfg)
    else
        error("未知训练方法: $(cfg.method)")
    end


    return initial_model
end




# ==============================================================================
# Random Forest 训练策略：稳定性选择 + 正向化对齐
# ==============================================================================
function _fit_rf_strategy(X, labels, gene_ids, pairs_idx, cfg)
    cfg.verbose && println(">>> 执行随机森林 (Stumps) 稳定性选择...")
    
	_, top_models = select_top_10_models(X, labels, cfg.ss_iterations, cfg.ss_ratio, cfg.target_n; verbose=cfg.verbose)
    isempty(top_models) && error("随机森林稳定性选择未得到有效模型，请检查样本量或放宽过滤参数。")

	final_model = top_models[1].model
    raw_weights = impurity_importance(final_model)


    # 4. 正向化对齐 (Orientation Alignment)
    # 调整基因对顺序，使得 g1 > g2 始终对应 Positive (1)
    final_named_pairs = Vector{Tuple{String, String}}()
    aligned_weights = Float64[]
    
	top_indices = unique([tree.featid for tree in final_model.trees if isa(tree, Node)])
	raw_weights = raw_weights[top_indices]
    for (i, idx) in enumerate(top_indices)
        g1_idx, g2_idx = pairs_idx[idx]
        p_rate = mean(X[labels .== 1, idx])
        n_rate = mean(X[labels .== 0, idx])
        
        if p_rate >= n_rate
            push!(final_named_pairs, (gene_ids[g1_idx], gene_ids[g2_idx]))
        else
            push!(final_named_pairs, (gene_ids[g2_idx], gene_ids[g1_idx]))
        end
        push!(aligned_weights, raw_weights[i])
    end
    

    # 5. 权重归一化与截距计算
    # RF 模式下，评分 = (满足对数权重和) / (总权重)
    # 此时 weights 之和为 1，截距设为 -0.5 实现 0.5 阈值判定
    weight_sum = sum(aligned_weights)
    norm_weights = weight_sum > 0 ? aligned_weights ./ weight_sum : fill(1.0 / length(aligned_weights), length(aligned_weights))
    intercept = 0
    
    return REOModel(cfg, final_named_pairs, norm_weights, intercept)
end


"""
    select_top_10_models(X, y, n_iterations=500, sub_sample_ratio=0.7, target_n=50; verbose=true)

重复训练随机森林树桩模型，并返回 OOB MCC 表现最好的模型集合及特征稳定性得分。

返回 `(selection_scores, top_models)`，其中 `selection_scores` 与特征列一一对应，
`top_models` 保存按 OOB MCC 排序后的最佳模型结果。
"""
function select_top_10_models(X, y, n_iterations=500, sub_sample_ratio=0.7, target_n=50; verbose=true)
    n_samples, n_features = size(X)
    idx1 = findall(==(1), y); idx0 = findall(==(0), y)
    
    # 临时存储所有模型的性能和对象
    # 使用 NamedTuple 存储：(model, train_mcc, oob_mcc)
    all_results = Vector{Any}(undef, n_iterations)
    verbose && println(">>> 开始迭代训练并筛选 Top 10 冠军模型...")

    Threads.@threads for i in 1:n_iterations
        # 1. 采样
        sub_idx = vcat(sample(idx1, Int(floor(length(idx1)*sub_sample_ratio)), replace=false),
                       sample(idx0, Int(floor(length(idx0)*sub_sample_ratio)), replace=false))
        oob_idx = setdiff(1:n_samples, sub_idx)

		# 2. 训练树桩森林(labels, features [AbstractMatrix{T}, n*p]
        model = build_forest(y[sub_idx], X[sub_idx, :], -1, target_n, 0.7, 1)

        # 3. 评估
        t_preds = apply_forest(model, X[sub_idx, :])
        o_preds = apply_forest(model, X[oob_idx, :])
        
        tmcc = _calculate_mcc(t_preds .> 0.5, y[sub_idx])
        omcc = _calculate_mcc(o_preds .> 0.5, y[oob_idx])

        # 存储结果
        all_results[i] = (model=model, train_mcc=tmcc, oob_mcc=omcc)
    end

    # --- 过滤无效结果 (如果有的话) 并排序 ---
    valid_results = filter(x -> !isnothing(x) && !isnan(x.oob_mcc), all_results)
    
    # 按 OOB MCC 从高到低排序
    sort!(valid_results, by = x -> x.oob_mcc, rev = true)

    # 取前 10 个
    top_10 = valid_results[1:min(10, length(valid_results))]

    # --- 计算基于这 10 个精英模型的特征稳定性得分 ---
    selection_scores = zeros(Float64, n_features)
    for res in top_10
        # 权重可以根据 OOB MCC 加权，也可以等权
        w = res.oob_mcc 
        for tree in res.model.trees
            if isa(tree, Node)
                selection_scores[tree.featid] += w
            end
        end
    end

    # 归一化得分
    max_s = maximum(selection_scores)
    final_scores = max_s > 0 ? selection_scores ./ max_s : selection_scores

    return final_scores, top_10
end


# ==============================================================================
# Lasso 训练策略：LassoPath 自动选择
# ==============================================================================

function _fit_lasso_strategy(X, labels, gene_ids, pairs_idx, cfg)
    cfg.verbose && println(">>> 执行 Lasso 路径 (含 OOB 加权稳定性选择)...")
    
    # 1. 加权稳定性选择
    sel_scores = stability_selection_lasso(X, labels, cfg)
    
    # 2. 筛选稳健特征
    stable_idx = findall(p -> p >= cfg.ss_threshold, sel_scores)
    if isempty(stable_idx)
        cfg.verbose && println("......未发现达到阈值的稳健特征，取前 $(cfg.target_n*2) 名...")
        stable_idx = sortperm(sel_scores, rev=true)[1:min(cfg.target_n*2, length(sel_scores))]
    end
    
    X_stable = X[:, stable_idx]
    pairs_stable = pairs_idx[stable_idx]

    # 3. 最终 Elastic Net 建模
    n_samples = length(labels)
    n_pos = sum(labels .== 1); n_neg = sum(labels .== 0)
    final_wts = [l == 1 ? (n_samples/(2*n_pos)) : (n_samples/(2*n_neg)) for l in labels]

    final_path = fit(LassoPath, X_stable, Float64.(labels), Binomial(), LogitLink(); 
                      wts=final_wts, standardize=true, intercept=true,
                      α=0.9, irls_maxiter=200, irls_tol=1e-4, λminratio=0.1)
    
    # 4. 提取参数
    raw_weights, raw_intercept, active_lasso_idx = extract_model_params(final_path, cfg.target_n)
    
    # 5. 正向化对齐 (Orientation Alignment)
    # 目标：调整基因对顺序使得所有权重均为正，支持 Positive 类
    final_named_pairs = Vector{Tuple{String, String}}()
    positive_weights = Float64[]
    adjusted_intercept = raw_intercept

    for i in 1:length(active_lasso_idx)
        idx_in_stable = active_lasso_idx[i]
        w = raw_weights[i]
        p = pairs_stable[idx_in_stable]
        g1_name, g2_name = gene_ids[p[1]], gene_ids[p[2]]
        
        if w > 0
            # 权重已经是正的，说明 g1 > g2 支持 Positive
            push!(final_named_pairs, (g1_name, g2_name))
            push!(positive_weights, w)
        else
            # 权重为负，说明 g1 > g2 支持 Negative，即 g2 > g1 支持 Positive
            # 翻转基因对，权重取绝对值
            # 逻辑：w*(g1 > g2) = w*(1 - (g2 > g1)) = -w*(g2 > g1) + w
            push!(final_named_pairs, (g2_name, g1_name))
            push!(positive_weights, abs(w))
            adjusted_intercept += w  # 截距修正
        end
    end
    
    # 归一化权重 (可选，为了与 RF 评分习惯保持一致)
    isempty(positive_weights) && error("Lasso 未选择任何有效基因对，请放宽稳定性选择参数。")
    total_w = sum(positive_weights)
    total_w == 0 && error("Lasso 选择的基因对权重和为 0，无法归一化。")
    final_weights = positive_weights ./ total_w
    final_intercept = adjusted_intercept / total_w
    
    return REOModel(cfg, final_named_pairs, final_weights, final_intercept)
end

"""
    extract_model_params(model_path, target_n)

从 Lasso 路径中提取非零特征数最接近 `target_n` 的系数、截距和活跃列索引。

返回 `(weights, intercept, active_idx)`，其中 `active_idx` 指向输入设计矩阵的列。
"""
function extract_model_params(model_path, target_n)
    coef_matrix = model_path.coefs
    # 计算每个 lambda 下非零个数
    n_nonzero = [count(!iszero, coef_matrix[:, i]) for i in 1:size(coef_matrix, 2)]
    
    # 寻找最接近且不超过 target_n 的位置，或者第一个达到 target_n 的位置
    best_idx = findfirst(x -> x >= target_n, n_nonzero)
    isnothing(best_idx) && (best_idx = size(coef_matrix, 2))
    
    full_weights = coef_matrix[:, best_idx]
    active_idx = findall(!iszero, full_weights)
    
    return Vector(full_weights[active_idx]), model_path.b0[best_idx], active_idx
end

"""
    stability_selection_lasso(X, y, cfg)

执行带 OOB MCC 加权的 Lasso 稳定性选择，返回归一化特征选择得分。

返回向量长度等于 `size(X, 2)`，取值越大表示该特征在下采样训练中越稳定。
"""
function stability_selection_lasso(X, y, cfg::REOConfig)
    n_samples, n_features = size(X)
    all_idx = 1:n_samples
    
    # 1. 获取两类索引
    idx1 = findall(x -> x == 1, y)
    idx0 = findall(x -> x == 0, y)
    
    # 每一类抽取的比例
    n1_sub = max(Int(floor(length(idx1) * cfg.ss_ratio)), 3)
    n0_sub = max(Int(floor(length(idx0) * cfg.ss_ratio)), 3)

    selection_scores = zeros(Float64, n_features)
    lk = ReentrantLock()

    cfg.verbose && println("开始加权稳定性选择 (OOB 评估模式，迭代次数: $(cfg.ss_iterations))...")
    
    Threads.@threads for i in 1:cfg.ss_iterations
        # --- 步骤 A: 分层抽样 (In-bag vs OOB) ---
        sub_idx1 = sample(idx1, n1_sub, replace=false)
        sub_idx0 = sample(idx0, n0_sub, replace=false)
        sub_idx = vcat(sub_idx1, sub_idx0)
        oob_idx = setdiff(all_idx, sub_idx)
        
        X_sub, y_sub = X[sub_idx, :], y[sub_idx]
        X_oob, y_oob = X[oob_idx, :], y[oob_idx]

        # --- 步骤 B: 过滤常数列 ---
        col_vars = var(X_sub, dims=1)
        active_cols = [c[2] for c in findall(v -> v > 1e-10, col_vars)]
        if length(active_cols) < 2 continue end
        X_filtered = X_sub[:, active_cols]
        
        # 计算子集权重处理不均衡
        n_s = length(y_sub)
        n_s1 = sum(y_sub .== 1); n_s0 = sum(y_sub .== 0)
        sub_wts = [l == 1 ? (n_s/(2*n_s1)) : (n_s/(2*n_s0)) for l in y_sub]

        try
            # --- 步骤 C: Lasso 拟合 (Elastic Net 模式) ---
            path = fit(LassoPath, X_filtered, Float64.(y_sub), Binomial(), LogitLink(); 
                       wts=sub_wts, standardize=true, intercept=true, α=0.9,
                       irls_maxiter=200, irls_tol=1e-4, λminratio=0.05)
            
            # 选取路径中间的解作为代表
            mid_idx = size(path.coefs, 2) ÷ 2
            coefs = path.coefs[:, mid_idx]
            intercept = path.b0[mid_idx]
            
            selected_local = findall(!iszero, coefs)
            if isempty(selected_local) continue end
            selected_global = active_cols[selected_local]

            # --- 步骤 D: OOB 性能评估 ---
            w_active = coefs[selected_local]
            X_oob_active = X_oob[:, selected_global]
            z = (X_oob_active * w_active) .+ intercept
            oob_preds = (1.0 ./ (1.0 .+ exp.(-z))) .> 0.5
            
            # 调用包内的 evaluate_mcc 或 _calculate_mcc
            oob_mcc = _calculate_mcc(oob_preds, y_oob)
            
            if oob_mcc > 0
                lock(lk) do
                    selection_scores[selected_global] .+= oob_mcc
                end
            end
        catch e
            continue
        end
    end
    
    max_score = maximum(selection_scores)
    return max_score > 0 ? selection_scores ./ max_score : selection_scores
end

# ==============================================================================
# Basic majority-voting strategy, no weights
# MajorityVote 训练策略：Gray code for <= 25 features and SFFS search <= 128 features
# ==============================================================================
function _fit_voting_strategy(X, labels, gene_ids, pairs_idx, cfg)
    cfg.verbose && println(">>> 执行多数投票 (Voting) 稳定性选择...")

	top_indices, err, p_val, tau = select_feature_subset(UInt8.(X), UInt8.(labels); verbose=cfg.verbose)
    cfg.verbose && println(">>> Best threshold for majority vote: $tau")
	
	n_top = length(top_indices)
    cfg.verbose && println(">>> $n_top gene pairs were selected.")

	final_named_pairs = Vector{Tuple{String, String}}()
	for idx in top_indices
		g1_idx, g2_idx = pairs_idx[idx]
		push!(final_named_pairs, (gene_ids[g1_idx], gene_ids[g2_idx]))
	end

    # 4. 等权赋值 (Voting 核心)
    # 投票法中，每个基因对的权重相同
    voting_weights = fill(1.0 / n_top, n_top)
	voting_bias    = (cld(n_top, 2) -tau)/n_top
	
    cfg.verbose && println(final_named_pairs)

    # 5. 初步构建模型 (截距暂设为 0，将在 fit_reo 中通过自适应阈值校准)
    return REOModel(cfg, final_named_pairs, voting_weights, voting_bias)

end
