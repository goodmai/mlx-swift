// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class QuantizationTests: XCTestCase {
    func testQuantizedLinearAppliesNVFP4GlobalScaleOnMetal() {
        let (input, quantizedWeight, scales, biases, globalScale, bias, expected) =
            Device.withDefaultDevice(.cpu) {
                MLXRandom.seed(7)
                let weight = MLXRandom.uniform(low: -2, high: 2, [32, 32])
                let input = MLXRandom.uniform(low: -1, high: 1, [4, 32])
                let globalScale = MLX.max(abs(weight)).asType(.float32)
                let bias = MLXRandom.uniform(low: -1, high: 1, [32])
                let (quantizedWeight, scales, biases) = MLX.quantized(
                    weight, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)
                let expected =
                    matmul(
                        input,
                        dequantized(
                            quantizedWeight, scales: scales, biases: biases, groupSize: 16, bits: 4,
                            mode: .nvfp4, globalScale: globalScale, dtype: .float32
                        ).T)
                    + bias
                eval(quantizedWeight, scales, expected)
                return (input, quantizedWeight, scales, biases, globalScale, bias, expected)
            }

        let actual = Device.withDefaultDevice(.gpu) {
            QuantizedLinear(
                weight: quantizedWeight, bias: bias, scales: scales, biases: biases,
                groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)(input)
        }

        assertEqual(actual, expected, rtol: 1e-5, atol: 1e-5)
    }

    func testQuantizedEmbeddingAppliesNVFP4GlobalScaleOnMetal() {
        let (
            input, indices, quantizedWeight, scales, biases, globalScale, expectedLinear,
            expectedLookup
        ) = Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(11)
            let weight = MLXRandom.uniform(low: -2, high: 2, [32, 32])
            let input = MLXRandom.uniform(low: -1, high: 1, [4, 32])
            let indices = MLXArray([0, 3, 7, 31])
            let globalScale = MLX.max(abs(weight)).asType(.float32)
            let (quantizedWeight, scales, biases) = MLX.quantized(
                weight, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)
            let dequantizedWeight = dequantized(
                quantizedWeight, scales: scales, biases: biases, groupSize: 16, bits: 4,
                mode: .nvfp4, globalScale: globalScale, dtype: .float32)
            let expectedLinear = matmul(input, dequantizedWeight.T)
            let expectedLookup = dequantized(
                quantizedWeight, scales: scales, biases: biases, groupSize: 16, bits: 4,
                mode: .nvfp4, globalScale: globalScale)[indices].asType(.bfloat16)
            eval(quantizedWeight, scales, expectedLinear, expectedLookup)
            return (
                input, indices, quantizedWeight, scales, biases, globalScale, expectedLinear,
                expectedLookup
            )
        }

        let (actualLinear, actualLookup) = Device.withDefaultDevice(.gpu) {
            let embedding = QuantizedEmbedding(
                weight: quantizedWeight, scales: scales, biases: biases,
                groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)
            return (embedding.asLinear(input), embedding(indices))
        }

        assertEqual(actualLinear, expectedLinear, rtol: 1e-5, atol: 1e-5)
        assertEqual(actualLookup, expectedLookup, rtol: 1e-5, atol: 1e-5)
    }

    func testQuantizedLinearShapeDesc() {
        let linear1 = Linear(512, 1024)
        let quantized1 = linear1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(
            quantized1.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
        let linear2 = Linear(1024, 512, bias: false)
        let quantized2 = linear2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(inputDimensions=1024, outputDimensions=512, bias=false)")
        let linear3 = Linear(512, 1024)
        let quantized3 = linear3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
    }

    func testQuantizedEmbeddingShapeDesc() {
        let embedding1 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized1 = embedding1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(quantized1.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
        let embedding2 = Embedding(embeddingCount: 1024, dimensions: 512)
        let quantized2 = embedding2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(embeddingCount=1024, dimensions=512)")
        let embedding3 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized3 = embedding3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
    }

    func testQuantizedLinearMxfp4DoesNotCreateAffineBiases() {
        let quantized = QuantizedLinear(64, 64, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertNil(quantized.biases)
    }

    func testQuantizedLinearStoresGlobalScale() {
        let globalScale = MLXArray(1.0, dtype: .float32)
        let quantized = QuantizedLinear(
            weight: MLXArray.zeros([8, 4], dtype: .uint32),
            bias: nil,
            scales: MLXArray.ones([8, 4], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            globalScale: globalScale)

        XCTAssertNotNil(quantized.globalScale)
        XCTAssertEqual(quantized.globalScale?.dtype, .float32)
        XCTAssertNotNil(quantized.parameters()["global_scale"])
        XCTAssertNil(quantized.parameters()["globalScale"])
    }

    func testQuantizedEmbeddingStoresGlobalScale() {
        let globalScale = MLXArray(1.0, dtype: .float32)
        let quantized = QuantizedEmbedding(
            weight: MLXArray.zeros([8, 2], dtype: .uint32),
            scales: MLXArray.ones([8, 2], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            globalScale: globalScale)

        XCTAssertNotNil(quantized.globalScale)
        XCTAssertEqual(quantized.globalScale?.dtype, .float32)
        XCTAssertNotNil(quantized.parameters()["global_scale"])
        XCTAssertNil(quantized.parameters()["globalScale"])
    }

    func testQuantizedGlobalScaleIsOptionalParameter() {
        let linear = QuantizedLinear(
            weight: MLXArray.zeros([8, 4], dtype: .uint32),
            bias: nil,
            scales: MLXArray.ones([8, 4], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4)
        let embedding = QuantizedEmbedding(
            weight: MLXArray.zeros([8, 2], dtype: .uint32),
            scales: MLXArray.ones([8, 2], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4)

        XCTAssertNil(linear.globalScale)
        XCTAssertNil(linear.parameters()["global_scale"])
        XCTAssertNil(embedding.globalScale)
        XCTAssertNil(embedding.parameters()["global_scale"])
    }

    // MARK: - nf4

    static let nf4Codebook: [Float] = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ]

    /// Reference NF4 quantizer written with plain MLX ops.
    static func nf4Reference(_ weight: MLXArray, groupSize: Int) -> (MLXArray, MLXArray) {
        let columns = weight.dim(-1)
        let groups = weight.asType(.float32).reshaped(-1, groupSize)
        let scales = abs(groups).max(axis: -1, keepDims: true).asType(.float16)
        let scale = scales.asType(.float32)
        let normalized = groups / which(scale .> 0, scale, 1)
        var codes = MLXArray.zeros(normalized.shape, dtype: .uint32)
        for (low, high) in zip(nf4Codebook, nf4Codebook.dropFirst()) {
            codes = codes + (normalized .> ((low + high) / 2)).asType(.uint32)
        }
        let shifts = MLXArray((0 ..< 8).map { UInt32(1) << UInt32(4 * $0) })
        let packed = (codes.reshaped(-1, 8) * shifts).sum(axis: -1).asType(.uint32)
        let leading = Array(weight.shape.dropLast())
        return (
            packed.reshaped(leading + [columns / 8]),
            scales.reshaped(leading + [columns / groupSize])
        )
    }

    static func nf4ReferenceDequantized(_ weight: MLXArray, scales: MLXArray, groupSize: Int)
        -> MLXArray
    {
        let shifts = MLXArray((0 ..< 8).map { UInt32(4 * $0) })
        let codes = (weight.expandedDimensions(axis: -1) >> shifts) & UInt32(15)
        let values = MLXArray(nf4Codebook).take(codes.reshaped(-1))
        let grouped = values.reshaped(-1, groupSize) * scales.reshaped(-1, 1).asType(.float32)
        return grouped.reshaped(Array(weight.shape.dropLast()) + [weight.dim(-1) * 8])
            .asType(.float16)
    }

    /// Normal weights with an all-zero group and a group whose entries sit exactly on the
    /// codebook midpoints.
    static func nf4TestWeight(rows: Int, columns: Int, dtype: DType) -> MLXArray {
        MLXRandom.seed(42)
        var weight = MLXRandom.normal([rows, columns]) * 0.05
        weight[0, 0 ..< 128] = MLXArray.zeros([128])
        let midpoints = zip(nf4Codebook, nf4Codebook.dropFirst()).map { ($0 + $1) / 2 }
        weight[1, 0 ..< 16] = MLXArray(midpoints + [1.0])
        return weight.asType(dtype)
    }

    func testNF4QuantizeMatchesReference() {
        for dtype in [DType.float32, .bfloat16, .float16] {
            for groupSize in [32, 64, 128] {
                let weight = Self.nf4TestWeight(rows: 64, columns: 1024, dtype: dtype)
                let (expectedCodes, expectedScales) = Self.nf4Reference(
                    weight, groupSize: groupSize)
                for device in [Device.gpu, .cpu] {
                    let (codes, scales, biases) = Device.withDefaultDevice(device) {
                        MLX.quantized(weight, groupSize: groupSize, bits: 4, mode: .nf4)
                    }
                    XCTAssertNil(biases)
                    XCTAssertEqual(codes.dtype, .uint32)
                    XCTAssertEqual(scales.dtype, .float16)
                    XCTAssertEqual(codes.shape, [64, 128])
                    XCTAssertEqual(scales.shape, [64, 1024 / groupSize])
                    XCTAssertTrue(
                        arrayEqual(codes, expectedCodes).item(Bool.self),
                        "codes \(dtype) g\(groupSize) \(device)")
                    XCTAssertTrue(
                        arrayEqual(scales, expectedScales).item(Bool.self),
                        "scales \(dtype) g\(groupSize) \(device)")
                }
            }
        }
    }

    func testNF4DequantizeRoundTrip() {
        for groupSize in [32, 64, 128] {
            let weight = Self.nf4TestWeight(rows: 64, columns: 1024, dtype: .float32)
            let (codes, scales, _) = MLX.quantized(
                weight, groupSize: groupSize, bits: 4, mode: .nf4)
            let expected = Self.nf4ReferenceDequantized(
                codes, scales: scales, groupSize: groupSize)
            for device in [Device.gpu, .cpu] {
                let restored = Device.withDefaultDevice(device) {
                    dequantized(
                        codes, scales: scales, biases: nil, groupSize: groupSize, bits: 4,
                        mode: .nf4)
                }
                XCTAssertEqual(restored.dtype, .float16)
                XCTAssertTrue(
                    arrayEqual(restored, expected).item(Bool.self), "g\(groupSize) \(device)")
            }
            let difference = weight - expected.asType(.float32)
            let relative = sqrt(sum(difference * difference) / sum(weight * weight))
                .item(Float.self)
            XCTAssertLessThan(relative, 0.12, "g\(groupSize)")
            XCTAssertTrue(arrayEqual(expected[0, 0 ..< 128], MLXArray.zeros([128])).item(Bool.self))
            XCTAssertEqual(
                dequantized(
                    codes, scales: scales, biases: nil, groupSize: groupSize, bits: 4,
                    mode: .nf4, dtype: .bfloat16
                ).dtype, .bfloat16)
        }
    }

    static func relativeMaxError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
        let expected = expected.asType(.float32)
        return (abs(actual.asType(.float32) - expected).max() / abs(expected).max())
            .item(Float.self)
    }

    func testNF4QuantizedMMMatchesDequantizedMatmul() {
        let shapes = [(k: 1024, n: 96), (k: 1536, n: 100), (k: 384, n: 64)]
        for dtype in [DType.bfloat16, .float16] {
            for groupSize in [32, 64, 128] {
                for (k, n) in shapes {
                    let weight = Self.nf4TestWeight(rows: n, columns: k, dtype: .float32)
                    let (codes, scales, _) = MLX.quantized(
                        weight, groupSize: groupSize, bits: 4, mode: .nf4)
                    let dense = dequantized(
                        codes, scales: scales, biases: nil, groupSize: groupSize, bits: 4,
                        mode: .nf4, dtype: .float32)
                    for m in [1, 2, 4, 8, 33] {
                        MLXRandom.seed(UInt64(m))
                        let x = MLXRandom.normal([m, k]).asType(dtype)
                        let expected = matmul(x.asType(.float32), dense.T)
                        let actual = quantizedMM(
                            x, codes, scales: scales, biases: nil, groupSize: groupSize, bits: 4,
                            mode: .nf4)
                        XCTAssertEqual(actual.dtype, dtype)
                        XCTAssertEqual(actual.shape, [m, n])
                        let error = Self.relativeMaxError(actual, expected)
                        XCTAssertLessThan(
                            error, 1e-2, "\(dtype) g\(groupSize) k\(k) n\(n) m\(m)")
                    }
                }
            }
        }
    }

    func testNF4QuantizedMMNotTransposed() {
        MLXRandom.seed(4)
        let weight = MLXRandom.normal([512, 256]) * 0.05
        for groupSize in [32, 64, 128] {
            let (codes, scales, _) = MLX.quantized(
                weight, groupSize: groupSize, bits: 4, mode: .nf4)
            let dense = dequantized(
                codes, scales: scales, biases: nil, groupSize: groupSize, bits: 4, mode: .nf4,
                dtype: .float32)
            for m in [1, 3, 33] {
                let x = MLXRandom.normal([m, 512]).asType(.float16)
                let expected = matmul(x.asType(.float32), dense)
                for device in [Device.gpu, .cpu] {
                    let actual = Device.withDefaultDevice(device) {
                        quantizedMM(
                            x, codes, scales: scales, biases: nil, transpose: false,
                            groupSize: groupSize, bits: 4, mode: .nf4)
                    }
                    XCTAssertEqual(actual.shape, [m, 256])
                    XCTAssertLessThan(
                        Self.relativeMaxError(actual, expected), 1e-2,
                        "g\(groupSize) m\(m) \(device)")
                }
            }
        }
    }

    func testNF4QuantizedMMBatchedAndCPU() {
        let weight = Self.nf4TestWeight(rows: 128, columns: 512, dtype: .float32)
        let (codes, scales, _) = MLX.quantized(weight, groupSize: 32, bits: 4, mode: .nf4)
        let dense = dequantized(
            codes, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .nf4,
            dtype: .float32)
        MLXRandom.seed(3)
        for m in [1, 4, 40] {
            let x = MLXRandom.normal([3, m, 512]).asType(.bfloat16)
            let expected = matmul(x.asType(.float32), dense.T)
            for device in [Device.gpu, .cpu] {
                let actual = Device.withDefaultDevice(device) {
                    quantizedMM(
                        x, codes, scales: scales, biases: nil, groupSize: 32, bits: 4,
                        mode: .nf4)
                }
                XCTAssertLessThan(Self.relativeMaxError(actual, expected), 1e-2, "m\(m) \(device)")
            }
        }
    }

    func testNF4GatherQuantizedMMMatchesDense() {
        let experts = 4
        let n = 96
        let k = 512
        MLXRandom.seed(5)
        let weight = MLXRandom.normal([experts, n, k]) * 0.05
        for groupSize in [32, 64] {
            let (codes, scales, _) = MLX.quantized(
                weight, groupSize: groupSize, bits: 4, mode: .nf4)
            let dense = dequantized(
                codes, scales: scales, biases: nil, groupSize: groupSize, bits: 4, mode: .nf4,
                dtype: .float32)
            for (batch, m, sorted) in [(6, 1, false), (32, 1, true), (5, 16, false), (3, 4, false)]
            {
                var indices = (0 ..< batch).map { UInt32(($0 * 7 + 3) % experts) }
                if sorted { indices.sort() }
                let rhs = MLXArray(indices)
                let x = MLXRandom.normal([batch, m, k]).asType(.bfloat16)
                let expected = matmul(
                    x.asType(.float32), dense[rhs].swappedAxes(-1, -2))
                for device in [Device.gpu, .cpu] {
                    let actual = Device.withDefaultDevice(device) {
                        gatherQuantizedMM(
                            x, codes, scales: scales, rhsIndices: rhs, groupSize: groupSize,
                            bits: 4, mode: .nf4, sortedIndices: sorted)
                    }
                    XCTAssertEqual(actual.shape, [batch, m, n])
                    XCTAssertLessThan(
                        Self.relativeMaxError(actual, expected), 1e-2,
                        "g\(groupSize) b\(batch) m\(m) sorted=\(sorted) \(device)")
                }
            }
        }
    }

    func testNF4RejectsInvalidParameters() {
        let (codes, scales, _) = MLX.quantized(
            MLXRandom.normal([8, 64]), groupSize: 32, bits: 4, mode: .nf4)
        XCTAssertThrowsError(
            try withError {
                _ = dequantized(
                    codes, scales: scales, biases: nil, groupSize: 16, bits: 4, mode: .nf4)
            })
        XCTAssertThrowsError(
            try withError {
                _ = dequantized(
                    codes, scales: scales.asType(.bfloat16), biases: nil, groupSize: 32,
                    bits: 4, mode: .nf4)
            })
        XCTAssertThrowsError(
            try withError {
                _ = quantizedMM(
                    MLXRandom.normal([1, 64]), codes, scales: scales, biases: scales,
                    groupSize: 32, bits: 4, mode: .nf4)
            })
    }

    func testAffineAndMxfp4MatmulStillMatchDequantized() {
        MLXRandom.seed(9)
        let weight = MLXRandom.normal([96, 1024]) * 0.05
        let configurations: [(QuantizationMode, Int)] = [
            (.affine, 64), (.affine, 32), (.mxfp4, 32),
        ]
        for (mode, groupSize) in configurations {
            let (codes, scales, biases) = MLX.quantized(
                weight, groupSize: groupSize, bits: 4, mode: mode)
            let dense = dequantized(
                codes, scales: scales, biases: biases, groupSize: groupSize, bits: 4, mode: mode,
                dtype: .float32)
            for m in [1, 2, 4, 8, 33] {
                let x = MLXRandom.normal([m, 1024]).asType(.bfloat16)
                let expected = matmul(x.asType(.float32), dense.T)
                let actual = quantizedMM(
                    x, codes, scales: scales, biases: biases, groupSize: groupSize, bits: 4,
                    mode: mode)
                XCTAssertLessThan(Self.relativeMaxError(actual, expected), 1e-2, "\(mode) m\(m)")
            }
        }
    }

    /// µs per `quantizedMM` call, nf4 g32 against affine 4-bit g64, on large projection
    /// shapes at small M. Set `MLX_NF4_BENCH=1` and build in release to run it.
    func testNF4MatmulBenchmark() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLX_NF4_BENCH"] == "1", "set MLX_NF4_BENCH=1")
        let samples = 7
        let cacheDefeatBytes = 256 << 20
        let shapes = [(5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120)]
        let configurations: [(QuantizationMode, Int)] = [(.nf4, 32), (.affine, 64)]

        func median(_ body: () -> Void) -> Double {
            let clock = ContinuousClock()
            let times = (0 ..< samples).map { _ in
                let elapsed = clock.measure(body)
                return Double(elapsed.components.seconds) * 1e6
                    + Double(elapsed.components.attoseconds) / 1e12
            }
            return times.sorted()[samples / 2]
        }

        print("K x N | M | nf4 g32 us | affine g64 us | nf4/affine")
        for (k, n) in shapes {
            MLXRandom.seed(1)
            let dense = MLXRandom.normal([n, k], dtype: .bfloat16) * 0.02
            var micros: [Int: [Double]] = [:]
            for (mode, groupSize) in configurations {
                let first = MLX.quantized(dense, groupSize: groupSize, bits: 4, mode: mode)
                eval([first.wq, first.scales] + (first.biases.map { [$0] } ?? []))
                let bytes = first.wq.nbytes + first.scales.nbytes + (first.biases?.nbytes ?? 0)
                var copies = [(first.wq, first.scales, first.biases)]
                for _ in 1 ..< max(2, cacheDefeatBytes / bytes + 1) {
                    let copy = (first.wq + 0, first.scales + 0, first.biases.map { $0 + 0 })
                    eval([copy.0, copy.1] + (copy.2.map { [$0] } ?? []))
                    copies.append(copy)
                }
                let rounds = max(1, 16 / copies.count)
                for m in [1, 2, 4, 8] {
                    let x = MLXRandom.normal([m, k], dtype: .bfloat16)
                    let run = {
                        var outputs: [MLXArray] = []
                        for _ in 0 ..< rounds {
                            for (w, scales, biases) in copies {
                                outputs.append(
                                    quantizedMM(
                                        x, w, scales: scales, biases: biases,
                                        groupSize: groupSize, bits: 4, mode: mode))
                            }
                        }
                        eval(outputs)
                    }
                    run()
                    micros[m, default: []].append(
                        median(run) / Double(rounds * copies.count))
                }
            }
            for m in [1, 2, 4, 8] {
                let (nf4, affine) = (micros[m]![0], micros[m]![1])
                print(
                    "\(k) x \(n) | \(m) | "
                        + String(format: "%.1f | %.1f | %.3f", nf4, affine, nf4 / affine))
            }
        }
    }
}
