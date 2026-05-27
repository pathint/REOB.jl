@testset "二分类评估指标" begin
    labels = [1, 1, 0, 0]

    @test REOB._calculate_mcc(Bool[1, 1, 0, 0], labels) == 1.0
    @test REOB._calculate_mcc(Bool[0, 0, 1, 1], labels) == -1.0
    @test REOB._calculate_mcc(Bool[1, 1, 1, 1], labels) == 0.0

    @test REOB._binary_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    @test REOB._binary_auc([0.5, 0.5, 0.5, 0.5], labels) == 0.5
    @test_throws ErrorException REOB._binary_auc([0.1], labels)
    @test_throws ErrorException REOB._binary_auc([0.1, 0.2], [1, 1])

    metrics = REOB._evaluate_binary_predictions(Bool[1, 0, 1, 0], labels)
    @test metrics.acc == 0.5
    @test metrics.mcc == 0.0
    @test metrics.preds == Bool[1, 0, 1, 0]
end
