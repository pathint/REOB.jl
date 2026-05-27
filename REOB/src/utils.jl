"""
    generate_test_data(n_genes=1000, n_samples=200; rng=Random.default_rng())

生成用于示例和单元测试的模拟 REO 数据。

返回 `(data, labels, genes)`：`data` 为基因 × 样本矩阵，`labels` 为
`0/1` 标签，`genes` 为基因名称。前两个基因被构造为稳定翻转的 REO
信号：正类中 `Gene_1 > Gene_2`，负类中 `Gene_1 < Gene_2`。
"""
function generate_test_data(n_genes::Int=1000, n_samples::Int=200; rng=Random.default_rng())
    n_genes >= 2 || error("至少需要 2 个基因才能生成 REO 测试数据。")
    n_samples >= 2 || error("至少需要 2 个样本才能生成 REO 测试数据。")

    data = randn(rng, n_genes, n_samples)
    n_pos = n_samples ÷ 2
    n_neg = n_samples - n_pos
    labels = vcat(ones(Int, n_pos), zeros(Int, n_neg))

    # 构造稳定翻转的序关系，便于 BQC 和 TSP 类方法识别。
    data[1, labels .== 1] .+= 2.0
    data[2, labels .== 1] .-= 2.0
    data[1, labels .== 0] .-= 2.0
    data[2, labels .== 0] .+= 2.0
    
    genes = ["Gene_$i" for i in 1:n_genes]
    return data, labels, genes
end
