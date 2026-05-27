using Statistics


"""
    fit_ktsp(data, labels, gene_names, cfg; k_max=9)

训练 k-Top Scoring Pairs (k-TSP) 模型。

候选基因先通过 `filter_genes` 预筛选，再按 TSP 分数贪心选择互不重叠的
基因对。最终 `k` 会调整为奇数以减少平局。

当预筛选后候选基因少于 2 个，或者最终没有选出任何有效基因对时会抛出错误。

# 返回值

返回 `KTSPModel`。
"""
function fit_ktsp(data::Matrix{T}, 
				  labels::AbstractVector, 
				  gene_names::Vector,
				  cfg::REOConfig; 
                  k_max=9) where T <: Real
    # --- 1. 预筛选：调用外部过滤函数，获取需要保留的基因索引 ---
	selected_genes = filter_genes(data, labels, gene_names, cfg)
    length(selected_genes) >= 2 || error("k-TSP 至少需要 2 个候选基因。")
	# 根据筛选结果裁剪数据和基因名
	data = data[selected_genes, :]          # 只保留选中基因的行
	gene_names = gene_names[selected_genes] # 对应的基因名
	n_genes, n_samples = size(data)         # 更新基因数量
    
	idx0 = findall(==(0), labels)
	idx1 = findall(==(1), labels)
    n0, n1 = length(idx0), length(idx1)

    # --- 1. 预筛选 (可选，提高效率) ---
    # 使用简单的 RankSum 或差异表达筛选前 N 个基因
    # 这里假设输入已经是预筛选过的，或者直接计算全部
    
    # --- 2. 计算所有对子的得分 ---
    all_scores = []
    # 为了避免重复计算和自比较，i < j
    for i in 1:n_genes-1
        for j in i+1:n_genes
            p0 = sum(data[i, idx0] .< data[j, idx0]) / n0
            p1 = sum(data[i, idx1] .< data[j, idx1]) / n1
            Δ = abs(p1 - p0)
            
            if Δ > 0
                # 二级评分：秩差之差 (Tie-breaker)
                # 为简化计算，这里仅在需要时计算，或作为三元组存储
                push!(all_scores, (i, j, Δ, p1 > p0))
            end
        end
    end

    # 按主评分 Δ 降序排列
    sort!(all_scores, by = x -> x[3], rev = true)

    # --- 3. 贪心选择互不重叠的对子 ---
    selected_pairs  = Tuple{Int, Int}[]
    selected_scores = Float64[]
    directions = Bool[]
    used_genes = Set{Int}()

    for (i, j, Δ, dir) in all_scores
        if length(selected_pairs) >= k_max
            break
        end
        
        # 核心约束：互不重叠 (Disjoint)
        if !(i in used_genes) && !(j in used_genes)
            push!(selected_pairs, (i, j))
            push!(selected_scores, Δ)
            push!(directions, dir)
            push!(used_genes, i)
            push!(used_genes, j)
        end
    end

    k_final = length(selected_pairs)
    k_final > 0 || error("k-TSP 未找到有效基因对。")
    # 强制 k 为奇数以避免平局（可选）
    if k_final % 2 == 0 && k_final > 0
        pop!(selected_pairs); pop!(selected_scores); pop!(directions)
        k_final -= 1
    end

    return KTSPModel(
        selected_pairs,
        [(gene_names[p[1]], gene_names[p[2]]) for p in selected_pairs],
        selected_scores,
        directions,
        k_final
    )
end


"""
    predict_ktsp(model, new_data, gene_names)

使用 k-TSP 模型进行多数投票预测。

若 `gene_names` 缺少模型所需基因，会抛出 `KeyError`。
"""
function predict_ktsp(model::KTSPModel, new_data::Matrix{T}, gene_names::Vector) where T <: Real
	# 构建基因名 → 行索引的映射
	gene_to_row = Dict(gene => i for (i, gene) in enumerate(gene_names))

    n_samples = size(new_data, 2)
    votes = zeros(Int, n_samples)

    for (idx, (gene_i, gene_j)) in enumerate(model.gene_names)
		i = gene_to_row[gene_i]
		j = gene_to_row[gene_j]
        is_less = new_data[i, :] .< new_data[j, :]
        # 如果方向为 true (p1 > p0)，则 Xi < Xj 投给 1
        if model.p_directions[idx]
            votes .+= is_less
        else
            votes .+= .!is_less
        end
    end

    return (votes .> (model.k / 2))
end
