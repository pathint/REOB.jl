using Statistics

"""
    fit_auctsp(data, labels, gene_names, cfg; k_max=9)

训练 AUC-TSP 模型。

算法为每个候选基因对计算两个方向的 AUC 分数，按 AUC 和二级差异分数排序，
再贪心选择互不重叠的基因对。

当预筛选后候选基因少于 2 个，或者最终没有选出任何有效基因对时会抛出错误。

# 返回值

返回 `AUCTSPModel`。
"""
function fit_auctsp(data::Matrix{T}, 
		labels::AbstractVector, 
		gene_names::Vector,                   
		cfg::REOConfig;
		k_max=9) where T <: Real
    # --- 1. 预筛选：调用外部过滤函数，获取需要保留的基因索引 ---
	selected_genes = filter_genes(data, labels, gene_names, cfg)
    length(selected_genes) >= 2 || error("AUC-TSP 至少需要 2 个候选基因。")
	# 根据筛选结果裁剪数据和基因名
	data = data[selected_genes, :]          # 只保留选中基因的行
	gene_names = gene_names[selected_genes] # 对应的基因名
    
	n_genes, n_samples = size(data)
    idx0 = findall(==(0), labels) # Control
    idx1 = findall(==(1), labels) # Case
    n0, n1 = length(idx0), length(idx1)

    all_pairs_stats = []

    # 1. 遍历所有对子计算 AUC
    # 复杂度 O(G^2)，建议输入前先做初步的差异表达筛选
    for i in 1:n_genes-1
        for j in i+1:n_genes
            # 计算序关系频率
            # p_ij_1: 在 Case 中 Xi < Xj 的比例
            p_ij_1 = sum(data[i, idx1] .< data[j, idx1]) / n1
            # p_ij_0: 在 Control 中 Xi < Xj 的比例
            p_ij_0 = sum(data[i, idx0] .< data[j, idx0]) / n0

            # AUCTSP 核心：计算两个方向的 AUC 并取最大值
            # 方向 A: Xi < Xj 预测为 1
            auc_a = (p_ij_1 + (1 - p_ij_0)) / 2
            # 方向 B: Xi > Xj 预测为 1
            auc_b = ((1 - p_ij_1) + p_ij_0) / 2
            
            best_auc = max(auc_a, auc_b)
            direction = auc_a >= auc_b ? 1 : -1 # 1 表示正向，-1 表示反向
            
            # 二级评分 (Tie-breaker): 使用 Wilcoxon 秩和统计量的绝对值
            # 论文建议在 AUC 相同时，选择组间秩差更显著的
            secondary = abs(mean(data[i, idx1] .- data[j, idx1]) - mean(data[i, idx0] .- data[j, idx0]))

            push!(all_pairs_stats, (i, j, best_auc, secondary, direction))
        end
    end

    # 2. 排序：首先按 AUC 降序，AUC 相同时按 secondary 降序
    sort!(all_pairs_stats, by = x -> (x[3], x[4]), rev = true)

    # 3. 贪心筛选互不重叠 (Disjoint) 的对子
    selected_pairs = []
    used_genes = Set{Int}()
    
    for (i, j, auc, _, dir) in all_pairs_stats
        if length(selected_pairs) >= k_max
            break
        end
        if !(i in used_genes) && !(j in used_genes)
            push!(selected_pairs, (i, j, auc, dir))
            push!(used_genes, i)
            push!(used_genes, j)
        end
    end

    k_final = length(selected_pairs)
    k_final > 0 || error("AUC-TSP 未找到有效基因对。")
    
    return AUCTSPModel(
        [(p[1], p[2]) for p in selected_pairs],
        [(gene_names[p[1]], gene_names[p[2]]) for p in selected_pairs],
        [p[3] for p in selected_pairs],
        [p[4] for p in selected_pairs],
        k_final
    )
end

"""
    predict_auctsp(model, new_data, gene_names)

使用 AUC-TSP 模型进行多数投票预测。

若 `gene_names` 缺少模型所需基因，会抛出 `KeyError`。
"""
function predict_auctsp(model::AUCTSPModel, new_data::Matrix{T}, gene_names::Vector) where T <: Real
	# 构建基因名 → 行索引的映射
	gene_to_row = Dict(gene => i for (i, gene) in enumerate(gene_names))

    n_samples = size(new_data, 2)
    votes = zeros(Float64, n_samples)

    for (idx, (name_i, name_j)) in enumerate(model.gene_names)
		i = gene_to_row[name_i]
		j = gene_to_row[name_j]
        # 根据训练时确定的方向进行投票
        if model.directions[idx] == 1
            votes .+= (new_data[i, :] .< new_data[j, :])
        else
            votes .+= (new_data[i, :] .> new_data[j, :])
        end
    end

    # 多数投票决策
	return (votes .> (model.k / 2))
end
