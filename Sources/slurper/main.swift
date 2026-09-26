import Foundation
import SlurperKit

let usage = """
    usage: slurper <youtube-url | audio-file> [--digitakt] [--kit STEMS] [--loops STEMS] [--bars N] [--bpm N] [--out DIR]
           slurper --version

    Writes the mix and its vocals, horns, drums, bass, other and instrumental stems as 44.1 kHz float WAVs.
      --digitakt     also write 48 kHz 16-bit copies for the Elektron Digitakt II
      --kit STEMS    also keep one example of each distinct hit in these stems
      --loops STEMS  also cut these stems into loops at the drums' bar lines, leaving out repeats
      --bars N       bars per loop (default 4)
      --bpm N        tempo for the bar lines, when the estimate lands on half or double
      --out DIR      parent folder for the stems (default ~/Music/Slurper/Stems)
    STEMS is a comma-separated list from \(StemSplitter.stemNames.joined(separator: ", ")).
    """

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(2)
}

var arguments = Array(CommandLine.arguments.dropFirst())
if arguments == ["--version"] {
    print("slurper \(StemSplitter.version)")
    exit(0)
}
var options = SplitOptions(digitakt: arguments.contains("--digitakt"))
arguments.removeAll { $0 == "--digitakt" }
var values: [String: String] = [:]
for flag in ["--out", "--kit", "--loops", "--bars", "--bpm"] {
    if let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) {
        values[flag] = arguments[index + 1]
        arguments.removeSubrange(index...index + 1)
    }
}
guard arguments.count == 1, !arguments[0].hasPrefix("-") else {
    FileHandle.standardError.write(Data((usage + "\n").utf8))
    exit(arguments == ["--help"] || arguments == ["-h"] ? 0 : 2)
}

@MainActor func stems(for flag: String) -> Set<String> {
    let stems = Set(values[flag]?.split(separator: ",").map(String.init) ?? [])
    if let unknown = stems.subtracting(StemSplitter.stemNames).sorted().first {
        fail("no stem named \"\(unknown)\"; \(flag) takes \(StemSplitter.stemNames.joined(separator: ", "))")
    }
    return stems
}
options.kit = stems(for: "--kit")
options.loops = stems(for: "--loops")
if let bars = values["--bars"] {
    guard let count = Int(bars), count > 0 else { fail("--bars takes a positive whole number, not \"\(bars)\"") }
    options.bars = count
}
if let bpm = values["--bpm"] {
    guard let tempo = Double(bpm), (20...400).contains(tempo) else { fail("--bpm takes a tempo from 20 to 400, not \"\(bpm)\"") }
    options.bpm = tempo
}
if options.loops.isEmpty, let flag = ["--bars", "--bpm"].first(where: { values[$0] != nil }) {
    fail("\(flag) only applies with --loops")
}
let output = values["--out"].map { URL(filePath: $0, directoryHint: .isDirectory) } ?? StemSplitter.outputRoot

let input = arguments[0]
let isFile = FileManager.default.fileExists(atPath: input)
if !isFile, URL(string: input)?.scheme == nil {
    fail("no such file: \(input)")
}
let events = isFile
    ? StemSplitter.split(file: URL(filePath: input), outputRoot: output, options: options)
    : StemSplitter.split(url: input, outputRoot: output, options: options)

let start = ContinuousClock.now
var lastDecile: [SplitEvent.Stage: Int] = [:]
do {
    for try await event in events {
        let decile = event.progress.map { Int($0 * 10) } ?? -1
        if lastDecile[event.stage] == decile { continue }
        lastDecile[event.stage] = decile
        let fields = [
            "\((ContinuousClock.now - start).components.seconds)s", event.stage.rawValue,
            event.progress.map { "\(Int($0 * 100))%" }, event.title, event.note, event.folder?.path,
        ].compactMap { $0 }
        print(fields.joined(separator: "  "))
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
    exit(1)
}
