@testset "传统 TSP 系列模型" begin
    data, labels, genes = generate_test_data(20, 40; rng=MersenneTwister(40))
    cfg = REOConfig(low_rank_q=0.0, top_diff_n=8, verbose=false)

    tsp = fit_tsp(data, labels, genes, cfg)
    @test tsp isa TSPModel
    @test Set(tsp.gene_names) == Set(("Gene_1", "Gene_2"))
    @test predict_tsp(tsp, data, genes) == Bool.(labels)
    @test evaluate_tsp(tsp, data, genes, labels).acc == 1.0
    @test_throws KeyError predict_tsp(tsp, data, ["Missing_$i" for i in eachindex(genes)])

    ktsp = fit_ktsp(data, labels, genes, cfg; k_max=3)
    @test ktsp isa KTSPModel
    @test isodd(ktsp.k)
    @test ktsp.k <= 3
    @test length(predict_ktsp(ktsp, data, genes)) == length(labels)
    @test evaluate_ktsp(ktsp, data, genes, labels).acc >= 0.8
    @test_throws KeyError predict_ktsp(ktsp, data, ["Missing_$i" for i in eachindex(genes)])

    auctsp = fit_auctsp(data, labels, genes, cfg; k_max=3)
    @test auctsp isa AUCTSPModel
    @test auctsp.k <= 3
    @test length(predict_auctsp(auctsp, data, genes)) == length(labels)
    @test evaluate_auctsp(auctsp, data, genes, labels).acc >= 0.8
    @test_throws KeyError predict_auctsp(auctsp, data, ["Missing_$i" for i in eachindex(genes)])

    one_gene_cfg = REOConfig(low_rank_q=0.0, top_diff_n=1, verbose=false)
    @test_throws ErrorException fit_tsp(data, labels, genes, one_gene_cfg)
    @test_throws ErrorException fit_ktsp(data, labels, genes, one_gene_cfg; k_max=3)
    @test_throws ErrorException fit_auctsp(data, labels, genes, one_gene_cfg; k_max=3)
end
