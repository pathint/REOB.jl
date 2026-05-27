# src/evaluation.jl

using Random, Statistics

"""
    predict_reo(model, test_data, test_gene_ids)

使用训练好的 `REOModel` 预测新样本。

# 参数

- `model`：由 [`fit_reo`](@ref) 返回的模型。
- `test_data`：基因 × 样本表达矩阵。
- `test_gene_ids`：与 `test_data` 行顺序一致的基因名称。

# 返回值

命名元组 `(probs, preds)`，其中 `probs` 为阳性概率或投票分数，
`preds` 为布尔预测标签。
"""
function predict_reo(model::REOModel, test_data::Matrix{<:Real}, test_gene_ids::Vector)
    n_samples = size(test_data, 2)
    gene_to_idx = Dict(name => i for (i, name) in enumerate(test_gene_ids))
    
    # 1. 筛选在测试集中存在的基因对
    available_weights = Float64[]
    active_pairs = []
    
    for (i, pair) in enumerate(model.final_pairs)
        g1_name, g2_name = pair
        if haskey(gene_to_idx, g1_name) && haskey(gene_to_idx, g2_name)
            push!(active_pairs, (gene_to_idx[g1_name], gene_to_idx[g2_name]))
            push!(available_weights, model.weights[i])
        end
    end
    
    if isempty(active_pairs)
        error("测试集与模型基因对完全不匹配，无法预测。")
    end

    # 2. 构建特征矩阵 X_val
    X_val = zeros(Float64, n_samples, length(active_pairs))
    for (j, (g1_idx, g2_idx)) in enumerate(active_pairs)
        @views X_val[:, j] .= test_data[g1_idx, :] .> test_data[g2_idx, :]
    end
    
    # 3. 计算原始线性得分 z
    z = (X_val * available_weights) .+ model.intercept
    
    # 4. 根据模型类型映射为 0-1 概率
    if model.config.method == LassoMethod
        # Lasso 模式：使用标准 Sigmoid (Logistic) 映射
        probs = 1.0 ./ (1.0 .+ exp.(-z))
    else
        # VotingMethod / RFMethod 模式：基于正向化加权投票。
        # 此时权重和为 1，z 的范围理论在 [0, 1] 之间 (假设 intercept = 0)
        # 我们将其平移回 0-1 区间
        probs = clamp.(z, 0, 1)
    end
    preds = probs .>= 0.5
    return (probs=probs, preds=preds)
end

"""
    evaluate_reo(model, valid_data, valid_gene_ids, valid_labels)

计算 REOB 模型的准确率、MCC、AUC 和预测结果。

返回 `(acc, mcc, auc, probs, preds)`。
"""
function evaluate_reo(model::REOModel, valid_data::Matrix{<:Real}, valid_gene_ids::Vector, valid_labels::AbstractVector)
    # 获取预测结果
    res = predict_reo(model, valid_data, valid_gene_ids)
    
    # 1. 准确率 (Accuracy)
    acc = mean(res.preds .== valid_labels)
    
    # 2. 马修斯相关系数 (MCC)
    mcc = _calculate_mcc(res.preds, valid_labels)
    
    # 3. AUC 计算
    auc_val = _binary_auc(res.probs, valid_labels)
    
    # 4. 混淆矩阵
    tp = sum((res.preds .== 1) .& (valid_labels .== 1))
    tn = sum((res.preds .== 0) .& (valid_labels .== 0))
    fp = sum((res.preds .== 1) .& (valid_labels .== 0))
    fn = sum((res.preds .== 0) .& (valid_labels .== 1))
    
    if model.config.verbose
        println("--- REO 模型评估报告 ($(model.config.method)) ---")
        println("特征对数量: $(length(model.final_pairs))")
        println("准确率 (ACC): $(round(acc, digits=4))")
        println("相关系数 (MCC): $(round(mcc, digits=4))")
        println("曲线下面积 (AUC): $(round(auc_val, digits=4))")
        println("混淆矩阵: [TP: $tp, FP: $fp; FN: $fn, TN: $tn]")
    end
    
    return (acc=acc, mcc=mcc, auc=auc_val, probs=res.probs, preds=res.preds)
end

"""
    _calculate_mcc(preds, labels)

计算二分类马修斯相关系数（MCC）。
"""
function _calculate_mcc(preds::AbstractVector{Bool}, labels::AbstractVector)
    tp = Float64(sum(preds .& (labels .== 1)))
    tn = Float64(sum(.!preds .& (labels .== 0)))
    fp = Float64(sum(preds .& (labels .== 0)))
    fn = Float64(sum(.!preds .& (labels .== 1)))
    
    num = (tp * tn) - (fp * fn)
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return den == 0 ? 0.0 : num / den
end

"""
    _binary_auc(scores, labels)

按秩和公式计算二分类 AUC。

`scores` 与 `labels` 长度必须一致，且标签必须同时包含 `0` 和 `1`。
"""
function _binary_auc(scores::AbstractVector{<:Real}, labels::AbstractVector)
    length(scores) == length(labels) || error("scores 和 labels 长度必须一致。")

    n = length(labels)
    pos_mask = labels .== 1
    neg_mask = labels .== 0
    n_pos = count(pos_mask)
    n_neg = count(neg_mask)
    (n_pos > 0 && n_neg > 0) || error("AUC 需要同时包含 0 和 1 标签。")

    order = sortperm(scores)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && scores[order[j + 1]] == scores[order[i]]
            j += 1
        end

        avg_rank = (i + j) / 2
        for k in i:j
            ranks[order[k]] = avg_rank
        end
        i = j + 1
    end

    pos_rank_sum = sum(ranks[pos_mask])
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
end


"""
    run_permutation_test(data, labels, genes, model, cfg; n_permutations=100)

通过打乱标签并重新拟合模型，估计观测 MCC 的置换检验 p 值。
"""
function run_permutation_test(data, labels, genes, model::REOModel, cfg::REOConfig; n_permutations=100)
    n_permutations > 0 || error("n_permutations 必须为正整数。")
    cfg.verbose && println(">>> 开始置换检验 ($n_permutations 次)...")
    
    # 1. 计算原始观测到的性能
    res = evaluate_reo(model, data, genes, labels)
    observed_mcc = res.mcc
    
    # 2. 迭代置换
    permuted_mccs = zeros(n_permutations)
    for i in 1:n_permutations
        shuffled_labels = shuffle(labels)
        try
            # 对打乱后的标签重新拟合模型
            p_model = fit_reo(data, shuffled_labels, genes, cfg)
            p_res = evaluate_reo(p_model, data, genes, shuffled_labels)
            permuted_mccs[i] = p_res.mcc
        catch
            permuted_mccs[i] = 0.0 # 若无法拟合则记为0
        end
    end
    
    p_value = (sum(permuted_mccs .>= observed_mcc) + 1) / (n_permutations + 1)
    return (p_value=p_value, observed_mcc=observed_mcc, permuted_mccs=permuted_mccs)
end

"""
    _evaluate_binary_predictions(preds, labels; title="", verbose=false)

计算二分类预测的准确率、MCC 和预测向量。
"""
function _evaluate_binary_predictions(preds::AbstractVector{Bool}, labels::AbstractVector; title::String="", verbose::Bool=false)
    acc = mean(preds .== labels)
    mcc = _calculate_mcc(preds, labels)

    tp = sum((preds .== 1) .& (labels .== 1))
    tn = sum((preds .== 0) .& (labels .== 0))
    fp = sum((preds .== 1) .& (labels .== 0))
    fn = sum((preds .== 0) .& (labels .== 1))

    if verbose
        println("--- $title 模型评估报告 ---")
        println("准确率 (ACC): $(round(acc, digits=4))")
        println("相关系数 (MCC): $(round(mcc, digits=4))")
        println("混淆矩阵: [TP: $tp, FP: $fp; FN: $fn, TN: $tn]")
    end

    return (acc=acc, mcc=mcc, preds=preds)
end

"""
    evaluate_tsp(model, valid_data, valid_gene_ids, valid_labels; verbose=false)

评估 TSP 模型，返回准确率、MCC 和预测标签。
"""
function evaluate_tsp(model::TSPModel, valid_data::Matrix{<:Real}, valid_gene_ids::Vector, valid_labels::AbstractVector; verbose::Bool=false)
    return _evaluate_binary_predictions(predict_tsp(model, valid_data, valid_gene_ids), valid_labels; title="TSP", verbose=verbose)
end

"""
    evaluate_ktsp(model, valid_data, valid_gene_ids, valid_labels; verbose=false)

评估 k-TSP 模型，返回准确率、MCC 和预测标签。
"""
function evaluate_ktsp(model::KTSPModel, valid_data::Matrix{<:Real}, valid_gene_ids::Vector, valid_labels::AbstractVector; verbose::Bool=false)
    return _evaluate_binary_predictions(predict_ktsp(model, valid_data, valid_gene_ids), valid_labels; title="k-TSP", verbose=verbose)
end

"""
    evaluate_auctsp(model, valid_data, valid_gene_ids, valid_labels; verbose=false)

评估 AUC-TSP 模型，返回准确率、MCC 和预测标签。
"""
function evaluate_auctsp(model::AUCTSPModel, valid_data::Matrix{<:Real}, valid_gene_ids::Vector, valid_labels::AbstractVector; verbose::Bool=false)
    return _evaluate_binary_predictions(predict_auctsp(model, valid_data, valid_gene_ids), valid_labels; title="AUC-TSP", verbose=verbose)
end
