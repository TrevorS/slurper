import AVFoundation
import Foundation
import Testing
@testable import SlurperKit

/// Runs the built `slurper` executable end to end.
struct CLITests {
    @Test(.enabled(if: FileManager.default.isExecutableFile(atPath: slurperBinary.path)))
    func printsUsageAndVersion() throws {
        let (missing, _, usage) = try run([])
        #expect(missing == 2)
        #expect(usage.hasPrefix("usage: slurper"))

        let (help, _, helpText) = try run(["--help"])
        #expect(help == 0)
        #expect(helpText.hasPrefix("usage: slurper"))

        let (version, output, _) = try run(["--version"])
        #expect(version == 0)
        #expect(output.hasPrefix("slurper \(StemSplitter.version)"), "\(output)")
    }

    @Test(.enabled(if: FileManager.default.isExecutableFile(atPath: slurperBinary.path)))
    func rejectsUnknownStemsAndBadNumbers() throws {
        let cases = [
            (["--kit", "drums,snare"], "\"snare\""), (["--loops", "drums", "--bars", "0"], "--bars"),
            (["--bpm", "fast"], "--bpm"), (["--bpm", "500"], "--bpm"),
            (["--bars", "8"], "--bars only applies with --loops"), ([], "no such file: song.wav"),
        ]
        for (arguments, complaint) in cases {
            let (status, _, errors) = try run(["song.wav"] + arguments)
            #expect(status == 2)
            #expect(errors.contains(complaint), "\(errors)")
        }
    }

    @Test(.enabled(if: modelsInstalled && FileManager.default.isExecutableFile(atPath: slurperBinary.path)))
    func splitsAFileIntoStems() throws {
        let directory = try temporaryDirectory()
        let input = directory.appending(path: "noise.wav")
        let frames = 44_100 * 12
        try AudioFiles.writeWAV(noise(frames: frames, channels: 2, amplitude: 0.1), sampleRate: 44_100, to: input)

        let arguments = [input.path, "--digitakt", "--kit", "drums", "--loops", "drums,bass", "--out", directory.appending(path: "out").path]
        let (status, output, errors) = try run(arguments)
        #expect(status == 0, "\(errors)")
        #expect(output.contains("done"))
        #expect(output.contains("  beats  "), "\(output)")

        let folder = directory.appending(path: "out/noise")
        for stem in StemSplitter.stemNames {
            let file = try AVAudioFile(forReading: folder.appending(path: "\(stem).wav"))
            #expect(file.length == AVAudioFramePosition(frames), "\(stem)")
            #expect(FileManager.default.fileExists(atPath: folder.appending(path: "Digitakt II/noise_\(stem).wav").path))
        }
    }

    private func run(_ arguments: [String]) throws -> (Int32, String, String) {
        let process = Process()
        process.executableURL = slurperBinary
        process.arguments = arguments
        let output = Pipe()
        let errors = Pipe()
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        let out = output.fileHandleForReading.readDataToEndOfFile()
        let err = errors.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(decoding: out, as: UTF8.self), String(decoding: err, as: UTF8.self))
    }
}
