import CoreML
import Foundation

/// A Core ML model package, compiled once next to itself and run on the GPU.
///
/// `MLModel` is not marked Sendable, but predictions on one instance are safe from several threads at once
/// (the vocal separator runs two chunks at a time on one model).
final class Model: @unchecked Sendable {
    private let model: MLModel

    /// Loads `package` (an `.mlpackage`), compiling it to a sibling `.mlmodelc` on the first run, and checks
    /// that it takes `inputs` and returns `outputs`.
    init(package: URL, inputs: [String], outputs: [String]) async throws {
        let compiled = package.deletingPathExtension().appendingPathExtension("mlmodelc")
        if !FileManager.default.fileExists(atPath: compiled.path) {
            let temporary = try await MLModel.compileModel(at: package)
            try FileManager.default.moveItem(at: temporary, to: compiled)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        model = try await MLModel.load(contentsOf: compiled, configuration: configuration)

        let description = model.modelDescription
        let missing = inputs.filter { description.inputDescriptionsByName[$0] == nil }
            + outputs.filter { description.outputDescriptionsByName[$0] == nil }
        guard missing.isEmpty else {
            throw SplitError.model("\(package.lastPathComponent) has no \(missing.joined(separator: ", "))")
        }
    }

    func run(_ inputs: [String: MLTensor]) async throws -> [String: MLTensor] {
        try await model.prediction(from: inputs)
    }
}

extension MLTensor {
    /// The tensor's values in row-major order.
    var values: [Float] {
        get async { await shapedArray(of: Float.self).scalars }
    }
}
