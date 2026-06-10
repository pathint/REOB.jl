using Statistics, StatsBase

"""
    fit_tsp(data, labels, gene_names, cfg)

训练 Top Scoring Pair (TSP) 模型。

# 参数

- `data`：基因 × 样本表达矩阵。
- `labels`：二分类标签，取值为 `0` 和 `1`。
- `gene_names`：与 `data` 行顺序一致的基因名称。
- `cfg`：用于预筛选基因的 `REOConfig`。

当预筛选后候选基因少于 2 个时会抛出错误。

# 返回值

返回 `TSPModel`。
"""
function fit_tsp(data::Matrix{T}, labels::AbstractVector, gene_names::Vector, cfg::REOConfig) where T <: Real
    # --- 1. 预筛选：调用外部过滤函数，获取需要保留的基因索引 ---
	selected_genes = filter_genes(data, labels, gene_names, cfg)
    length(selected_genes) >= 2 || error("TSP 至少需要 2 个候选基因。")
	# 根据筛选结果裁剪数据和基因名
	data = data[selected_genes, :]          # 只保留选中基因的行
	gene_names = gene_names[selected_genes] # 对应的基因名
	n_genes, n_samples = size(data)         # 更新基因数量
    
    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    
    n0 = length(idx0)
    n1 = length(idx1)

    best_score     = -1.0
    best_secondary = -1.0
    best_pair      = (0, 0)
    
    # 预计算所有基因在各组中的秩 (用于二级评分/Tie-breaker)
    # 论文建议在得分相同时，选择组间秩差(Rank Difference)最大的对子
    ranks = zeros(Float64, n_genes, n_samples)
    for j in 1:n_samples
        ranks[:, j] .= tiedrank(data[:, j])
    end

    # 遍历所有基因对 (i, j)
    # TSP 复杂度为 O(G^2 * S)，建议对基因进行初步筛选
    for i in 1:n_genes-1
        for j in i+1:n_genes
            # 计算 P(Xi < Xj | Class 0) 和 P(Xi < Xj | Class 1)
            k0 = sum(data[i, idx0] .< data[j, idx0])
            k1 = sum(data[i, idx1] .< data[j, idx1])
            
            p0 = k0 / n0
            p1 = k1 / n1
            
            # 主评分: Delta = |p1 - p0|
            Δ = abs(p1 - p0)
            
            if Δ > best_score
                best_score = Δ
                best_pair = (i, j)
                # 计算二级评分: 组间平均秩差之差
                best_secondary = abs(mean(ranks[i, idx1] .- ranks[j, idx1]) - 
                                     mean(ranks[i, idx0] .- ranks[j, idx0]))
            elseif Δ == best_score && Δ > 0
                # Tie-breaking 逻辑
                secondary = abs(mean(ranks[i, idx1] .- ranks[j, idx1]) - 
                                mean(ranks[i, idx0] .- ranks[j, idx0]))
                if secondary > best_secondary
                    best_secondary = secondary
                    best_pair = (i, j)
                end
            end
        end
    end

    i, j = best_pair
    # 计算最终模型的 p1, p2 用于预测方向
    p0_final = mean(data[i, idx0] .< data[j, idx0])
    p1_final = mean(data[i, idx1] .< data[j, idx1])

    return TSPModel(i, j, (gene_names[i], gene_names[j]), best_score, p0_final, p1_final)
end

"""
    predict_tsp(model, new_data, gene_names)

使用 TSP 模型预测新样本，返回布尔预测标签。

若 `gene_names` 缺少模型所需基因，会抛出 `KeyError`。
"""
function predict_tsp(model::TSPModel, new_data::Matrix{T}, gene_names::Vector) where T <: Real
	gene_to_row = Dict(gene => i for (i, gene) in enumerate(gene_names))
	name1, name2 = model.gene_names
	gene_i = gene_to_row[name1]
	gene_j = gene_to_row[name2]
    # 如果 p1 > p0，说明 Xi < Xj 指向 Class 1
    # 反之，说明 Xi < Xj 指向 Class 0
	
    is_less = new_data[gene_i, :] .< new_data[gene_j, :]
    
    if model.p1 > model.p0
		return (is_less, is_less)  # true -> 1, false -> 0
    else
        return (.!is_less, .!is_less) # true -> 0, false -> 1
    end
end
