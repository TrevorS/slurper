import Accelerate
import Foundation

public struct SplitEvent: Sendable {
    public enum Stage: String, Sendable {
        case models, download, decode, vocals, stems, beats, write, done
    }

    public var stage: Stage
    public var progress: Double? = nil
    public var title: String? = nil
    public var folder: URL? = nil
    public var note: String? = nil
}

public struct SplitOptions: Sendable {
    /// Also write 48 kHz 16-bit copies for the Digitakt II.
    public var digitakt: Bool
    /// Stems (from `StemSplitter.stemNames`) to build a kit from: one example of each distinct hit.
    public var kit: Set<String>
    /// Stems to cut into loops at the drum stem's bar lines.
    public var loops: Set<String>
    public var bars: Int
    /// Tempo for the loops' bar lines instead of estimating it.
    public var bpm: Double?

    public init(digitakt: Bool = false, kit: Set<String> = [], loops: Set<String> = [], bars: Int = 4, bpm: Double? = nil) {
        (self.digitakt, self.kit, self.loops, self.bars, self.bpm) = (digitakt, kit, loops, bars, bpm)
    }
}

enum SplitError: LocalizedError {
    case tool(String)
    case model(String)
    case audio(String)
    case download(String)

    var errorDescription: String? {
        switch self {
        case .tool(let message): message
        case .model(let message): "Model error: \(message)"
        case .audio(let message): "Audio error: \(message)"
        case .download(let message): "Model download failed: \(message)"
        }
    }
}

/// yt-dlp fetches the audio (or a local file is decoded), Mel-Band RoFormer on Core ML separates the vocals,
/// and htdemucs on Core ML splits the instrumental (mix minus vocals) into drums, bass and other.
public enum StemSplitter {
    public static let version = "0.1.0"
    public static let outputRoot = URL.musicDirectory.appending(path: "Slurper/Stems", directoryHint: .isDirectory)
    public static let stemNames = ["mix", "vocals", "instrumental", "drums", "bass", "other"]
    static let modelsDirectory = URL.applicationSupportDirectory.appending(path: "Slurper/Models", directoryHint: .isDirectory)
    static let vocalModel = modelsDirectory.appending(path: "MelBandRoformer-Vocal-CoreML/mbr_fp16.mlpackage", directoryHint: .isDirectory)
    static let demucsModel = modelsDirectory.appending(path: "htdemucs-CoreML/htdemucs_fp32.mlpackage", directoryHint: .isDirectory)

    static let roformerRepo = "TrevorJS/MelBandRoformer-Vocal-CoreML"
    static let roformerRevision = "498dbf1b3c800a72be07ab0b15ed37f9d2b2bb05"
    static let demucsRepo = "TrevorJS/htdemucs-CoreML"
    static let demucsRevision = "f46494c39557da0b318e8e33af2acc8b354504f6"

    /// Two concurrent chunks were fastest on an M2 (vocals 29 s vs 31 s for one, on a 60 s clip);
    /// three and four were slower.
    private static let vocalChunksAtOnce = 2

    private struct ModelFile {
        let repo: String
        let revision: String
        let folder: String
        let path: String
        /// Approximate size, used to weight download progress.
        let bytes: Int64

        var remote: URL { URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")! }
        var local: URL { StemSplitter.modelsDirectory.appending(path: "\(folder)/\(path)") }
    }

    /// An .mlpackage is a manifest, the model program and its weights. The vocal model's folder also holds
    /// the golden chunk the tests compare against.
    private static let modelFiles = [
        ModelFile(repo: roformerRepo, revision: roformerRevision, folder: "MelBandRoformer-Vocal-CoreML",
                  path: "mbr_fp16.mlpackage/Manifest.json", bytes: 617),
        ModelFile(repo: roformerRepo, revision: roformerRevision, folder: "MelBandRoformer-Vocal-CoreML",
                  path: "mbr_fp16.mlpackage/Data/com.apple.CoreML/model.mlmodel", bytes: 598_363),
        ModelFile(repo: roformerRepo, revision: roformerRevision, folder: "MelBandRoformer-Vocal-CoreML",
                  path: "mbr_fp16.mlpackage/Data/com.apple.CoreML/weights/weight.bin", bytes: 489_706_048),
        ModelFile(repo: roformerRepo, revision: roformerRevision, folder: "MelBandRoformer-Vocal-CoreML",
                  path: "golden_raw.f32", bytes: 2_822_400),
        ModelFile(repo: roformerRepo, revision: roformerRevision, folder: "MelBandRoformer-Vocal-CoreML",
                  path: "golden_vocals.f32", bytes: 2_822_400),
        ModelFile(repo: demucsRepo, revision: demucsRevision, folder: "htdemucs-CoreML",
                  path: "htdemucs_fp32.mlpackage/Manifest.json", bytes: 617),
        ModelFile(repo: demucsRepo, revision: demucsRevision, folder: "htdemucs-CoreML",
                  path: "htdemucs_fp32.mlpackage/Data/com.apple.CoreML/model.mlmodel", bytes: 538_090),
        ModelFile(repo: demucsRepo, revision: demucsRevision, folder: "htdemucs-CoreML",
                  path: "htdemucs_fp32.mlpackage/Data/com.apple.CoreML/weights/weight.bin", bytes: 209_160_960),
    ]

    /// yt-dlp needs ffmpeg and deno, which may not be on the caller's PATH.
    private static let searchPath = [
        "/opt/homebrew/bin", "/usr/local/bin", "\(NSHomeDirectory())/.local/bin", "/usr/bin", "/bin",
    ]

    public static func split(
        url: String, outputRoot: URL = outputRoot, options: SplitOptions = SplitOptions()
    ) -> AsyncThrowingStream<SplitEvent, Error> {
        stream { emit in
            let work = FileManager.default.temporaryDirectory.appending(path: "slurper-\(UUID().uuidString)", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: work) }

            async let models = loadModels(emit)
            async let source = fetch(url, into: work, emit: emit)
            let (title, mix) = try await source
            try await separate(mix, title: title, models: models, outputRoot: outputRoot, options: options, emit: emit)
        }
    }

    public static func split(
        file: URL, outputRoot: URL = outputRoot, options: SplitOptions = SplitOptions()
    ) -> AsyncThrowingStream<SplitEvent, Error> {
        stream { emit in
            let title = file.deletingPathExtension().lastPathComponent
            async let models = loadModels(emit)
            async let mix = decode(file, title: title, emit: emit)
            try await separate(mix, title: title, models: models, outputRoot: outputRoot, options: options, emit: emit)
        }
    }

    private static func stream(
        _ body: @escaping @Sendable (_ emit: @escaping @Sendable (SplitEvent) -> Void) async throws -> Void
    ) -> AsyncThrowingStream<SplitEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await body { continuation.yield($0) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Pipeline

    struct Models: Sendable {
        let vocals: VocalSeparator
        let demucs: Demucs
    }

    private static func loadModels(_ emit: @escaping @Sendable (SplitEvent) -> Void) async throws -> Models {
        try await downloadModels(emit)
        async let vocals = VocalSeparator(model: vocalModel, chunksAtOnce: vocalChunksAtOnce)
        async let demucs = Demucs(model: demucsModel)
        return try await Models(vocals: vocals, demucs: demucs)
    }

    private static func fetch(_ url: String, into work: URL, emit: @escaping @Sendable (SplitEvent) -> Void) async throws -> (String, [[Float]]) {
        emit(SplitEvent(stage: .download, progress: 0))
        let (title, file) = try await download(url, into: work, emit: emit)
        return (title, try await decode(file, title: title, emit: emit))
    }

    private static func decode(_ file: URL, title: String, emit: @Sendable (SplitEvent) -> Void) async throws -> [[Float]] {
        emit(SplitEvent(stage: .decode, title: title))
        return try AudioFiles.readStereo(file, sampleRate: Double(VocalSeparator.sampleRate))
    }

    /// Each stem is written (and exported for the Digitakt) as soon as it exists, while the next model runs.
    /// Stems cut into loops wait for the drum stem's bar lines.
    private static func separate(
        _ mix: [[Float]], title: String, models: Models, outputRoot: URL, options: SplitOptions,
        emit: @escaping @Sendable (SplitEvent) -> Void
    ) async throws {
        let output = try Output(folder: uniqueFolder(in: outputRoot, named: title), title: title, options: options)
        // A failed split leaves no partial folder, so running the song again reuses its name.
        var finished = false
        defer { if !finished { try? FileManager.default.removeItem(at: output.folder) } }
        try await withThrowingTaskGroup(of: Void.self) { writes in
            var waiting: [(name: String, audio: [[Float]])] = []
            func store(_ name: String, _ audio: [[Float]]) {
                if options.loops.contains(name) {
                    waiting.append((name, audio))
                } else {
                    writes.addTask { try output.save(name, audio) }
                }
            }
            store("mix", mix)

            emit(SplitEvent(stage: .vocals, progress: 0, title: title))
            let vocals = try await models.vocals.vocals(of: mix) { emit(SplitEvent(stage: .vocals, progress: $0, title: title)) }
            let instrumental = (0..<2).map { vDSP.subtract(mix[$0], vocals[$0]) }
            store("vocals", vocals)
            store("instrumental", instrumental)

            emit(SplitEvent(stage: .stems, progress: 0, title: title))
            let stems = try await models.demucs.stems(of: instrumental) { emit(SplitEvent(stage: .stems, progress: $0, title: title)) }
            for name in ["drums", "bass", "other"] {
                guard let stem = stems[name] else { throw SplitError.model("htdemucs returned no \(name) stem") }
                store(name, stem)
            }

            if !waiting.isEmpty, let drums = stems["drums"] {
                let grid = Beats.grid(of: drums, sampleRate: Double(VocalSeparator.sampleRate), bpm: options.bpm)
                let note = grid.map { grid in
                    let found = "\(Int(grid.bpm.rounded())) bpm, \(grid.downbeats.count) bar lines"
                    return grid.downbeats.count > options.bars ? found : found + ", too few for \(options.bars)-bar loops"
                } ?? "no steady beat in the drums, so no loops"
                emit(SplitEvent(stage: .beats, title: title, note: note))
                for (name, audio) in waiting { writes.addTask { try output.save(name, audio, grid: grid) } }
            }

            emit(SplitEvent(stage: .write, title: title))
            try await writes.waitForAll()
        }
        finished = true
        emit(SplitEvent(stage: .done, title: title, folder: output.folder))
    }

    private struct Output: Sendable {
        let folder: URL
        let title: String
        let options: SplitOptions

        /// Writes the stem, its kit if asked for, its loops if there is a `grid`, and their Digitakt copies.
        func save(_ name: String, _ audio: [[Float]], grid: Beats.Grid? = nil) throws {
            let rate = Double(VocalSeparator.sampleRate)
            try AudioFiles.writeWAV(audio, sampleRate: rate, to: folder.appending(path: "\(name).wav"))
            var chops: [Chops] = []
            if options.kit.contains(name) { chops.append(Kit.chops(of: audio, sampleRate: rate)) }
            if let grid { chops.append(grid.loops(of: audio, bars: options.bars, sampleRate: rate)) }
            for set in chops { try set.write(audio, sampleRate: rate, prefix: name, in: folder) }
            if options.digitakt { try exportDigitakt(name, audio, title: title, sampleRate: rate, chops: chops, to: folder) }
        }
    }

    /// Digitakt II plays 16-bit / 48 kHz mono or stereo WAVs natively (manual OS 1.16 p.29), so
    /// Elektron Transfer imports these without converting. Each stem is one file at full length;
    /// Elektronauts users report imports failing above ~59 MB (about 5 min stereo), which Elektron
    /// does not document. `chops` (sample ranges at `sampleRate`) are cut from the 48 kHz copy into
    /// `<prefix>_<stem>_<folder>/`, so they share the stem's mono or stereo choice.
    static func exportDigitakt(
        _ name: String, _ audio: [[Float]], title: String, sampleRate: Double, chops: [Chops] = [], to folder: URL
    ) throws {
        let directory = folder.appending(path: "Digitakt II", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var output = try AudioFiles.resample(audio, from: sampleRate, to: 48_000)
        if output.count == 2, vDSP.maximumMagnitude(vDSP.subtract(output[0], output[1])) < 1e-4 {
            output = [vDSP.multiply(0.5, vDSP.add(output[0], output[1]))]
        }
        let file = "\(digitaktPrefix(title))_\(name)"
        try AudioFiles.writeWAV(output, sampleRate: 48_000, encoding: .int16Dithered, to: directory.appending(path: "\(file).wav"))
        for set in chops {
            try set.scaled(by: 48_000 / sampleRate, frames: output[0].count)
                .write(output, sampleRate: 48_000, encoding: .int16Dithered, prefix: file, in: directory)
        }
    }

    /// First 20 characters of the title's ASCII words joined by underscores.
    static func digitaktPrefix(_ title: String) -> String {
        let words = title.unicodeScalars
            .map { $0.isASCII && CharacterSet.alphanumerics.contains($0) ? Character($0) : " " }
            .split(separator: " ")
        return words.isEmpty ? "song" : String(words.joined(separator: "_").prefix(20))
    }

    // MARK: - Download

    enum YTDLPLine: Equatable {
        case title(String)
        case file(String)
        case progress(Double)
    }

    /// Parses the tab-separated lines our `--print` and `--progress-template` arguments produce.
    static func parse(ytdlpLine line: String) -> YTDLPLine? {
        let fields = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        guard fields.count > 1 else { return nil }
        switch fields[0] {
        case "slurper-title":
            return .title(fields[1])
        case "slurper-file":
            return .file(fields[1])
        case "slurper-progress":
            guard fields.count > 2, let done = Double(fields[1]), let total = Double(fields[2]), total > 0 else { return nil }
            return .progress(min(done / total, 1))
        default:
            return nil
        }
    }

    private static func download(
        _ url: String, into work: URL, emit: @Sendable (SplitEvent) -> Void
    ) async throws -> (title: String, file: URL) {
        guard let ytdlp = tool("yt-dlp") else {
            throw SplitError.tool("yt-dlp not found. Install it with `brew install yt-dlp` (it also uses ffmpeg and deno).")
        }
        let arguments = [
            "--no-playlist", "--no-simulate", "--progress", "--newline",
            "--format", "bestaudio/best", "--extract-audio", "--audio-format", "wav",
            "--postprocessor-args", "ExtractAudio:-c:a pcm_f32le",
            "--output", work.appending(path: "source.%(ext)s").path,
            "--print", "before_dl:slurper-title\t%(title)s",
            "--print", "after_move:slurper-file\t%(filepath)s",
            "--progress-template", "download:slurper-progress\t%(progress.downloaded_bytes)s\t%(progress.total_bytes,progress.total_bytes_estimate)s",
            url,
        ]

        var title = "untitled"
        var file: URL?
        try await run(ytdlp, arguments) { line in
            switch parse(ytdlpLine: line) {
            case .title(let value): title = value
            case .file(let path): file = URL(filePath: path)
            case .progress(let fraction): emit(SplitEvent(stage: .download, progress: fraction, title: title))
            case nil: break
            }
        }
        guard let file else { throw SplitError.tool("yt-dlp finished without reporting the audio file") }
        return (title, file)
    }

    private static func downloadModels(_ emit: @escaping @Sendable (SplitEvent) -> Void) async throws {
        let missing = modelFiles.filter { !FileManager.default.fileExists(atPath: $0.local.path) }
        guard !missing.isEmpty else { return }
        let total = Double(missing.reduce(0) { $0 + $1.bytes })
        emit(SplitEvent(stage: .models, progress: 0))
        try await withThrowingTaskGroup(of: Int64.self) { group in
            for file in missing {
                group.addTask {
                    let (temporary, response) = try await URLSession.shared.download(from: file.remote)
                    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                    guard status == 200 else { throw SplitError.download("\(file.path): HTTP \(status)") }
                    let fm = FileManager.default
                    try fm.createDirectory(at: file.local.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try? fm.removeItem(at: file.local)
                    try fm.moveItem(at: temporary, to: file.local)
                    return file.bytes
                }
            }
            var done: Int64 = 0
            for try await bytes in group {
                done += bytes
                emit(SplitEvent(stage: .models, progress: Double(done) / total))
            }
        }
    }

    // MARK: - Helpers

    private static func tool(_ name: String) -> URL? {
        searchPath.lazy
            .map { URL(filePath: $0).appending(path: name) }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    /// Runs a command-line tool, handing each output line (stdout and stderr) to `onLine`.
    private static func run(_ executable: URL, _ arguments: [String], onLine: (String) -> Void) async throws {
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = (searchPath + [environment["PATH"]].compactMap { $0 }).joined(separator: ":")
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()

        var tail: [String] = []
        try await withTaskCancellationHandler {
            for try await line in pipe.fileHandleForReading.bytes.lines {
                onLine(line)
                tail.append(line)
                if tail.count > 20 { tail.removeFirst() }
            }
        } onCancel: {
            process.terminate()
        }
        process.waitUntilExit()
        try Task.checkCancellation()
        guard process.terminationStatus == 0 else {
            let reason = tail.last { $0.hasPrefix("ERROR") } ?? tail.last ?? "exit status \(process.terminationStatus)"
            throw SplitError.tool("\(executable.lastPathComponent): \(reason)")
        }
    }

    static func uniqueFolder(in root: URL, named title: String) throws -> URL {
        let cleaned = String(title.map { "/:\\".contains($0) || $0.isNewline ? "-" : $0 })
            .trimmingCharacters(in: .whitespaces.union(CharacterSet(charactersIn: ".")))
        let name = cleaned.isEmpty ? "untitled" : cleaned
        var folder = root.appending(path: name, directoryHint: .isDirectory)
        var n = 2
        while FileManager.default.fileExists(atPath: folder.path) {
            folder = root.appending(path: "\(name) \(n)", directoryHint: .isDirectory)
            n += 1
        }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }
}
