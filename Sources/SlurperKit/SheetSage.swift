import Accelerate
import CoreML
import Foundation

/// SheetSage2 (m-a-p; CC BY-NC 4.0 weights) on Core ML: beats and meter, sections, key, chords and the melody,
/// from 24 kHz mono audio.
///
/// The song is read in 300 s windows that start 100 s apart (`windows`). For each window the encoder turns the
/// zero-padded audio into the decoder's cross-attention keys and values, which this host copies into the
/// decoder's state; the decoder then runs one token at a time, each step's logits masked by the event grammar
/// (`SheetSageTokens.Grammar`) before the argmax. A window after the first starts from the events already
/// accepted in its first 100 s, re-encoded as a prefix, and each keeps the events between where the last one
/// stopped and 100 s before its end (all of the last window). This follows SheetSage2's
/// `pipeline_sheetsage2.py` with its default preset; the models come from `scripts/convert_sheetsage2.py`.
final class SheetSage: Sendable {
    static let sampleRate = 24_000.0
    static let windowSeconds = 300.0
    static let windowSamples = 7_200_000
    static let maxTokens = 5120
    private static let layers = 6
    private static let frames = 7500
    private static let headWidth = 64
    private static let heads = 8

    private let encoder: Model
    private let decoder: Model

    init(encoder: URL, decoder: URL) async throws {
        async let encoderModel = Model(package: encoder, inputs: ["samples"], outputs: ["cross"])
        async let decoderModel = Model(package: decoder, inputs: ["token", "position", "mask"], outputs: ["logits"])
        (self.encoder, self.decoder) = try await (encoderModel, decoderModel)
    }

    /// A stereo mix at `sampleRate` as the mono 24 kHz audio the model takes.
    static func mono(_ audio: [[Float]], sampleRate rate: Double) throws -> [Float] {
        try AudioFiles.resample([Transients.mono(audio)], from: rate, to: sampleRate)[0]
    }

    func transcribe(_ audio: [Float], progress: (Double) -> Void) async throws -> Transcription {
        let duration = Double(audio.count) / Self.sampleRate
        let plan = Self.windows(duration: duration)
        var stitched: [SheetSageTokens.Event] = []
        for (index, window) in plan.enumerated() {
            var prefix = SheetSageTokens.prompt
            var base = 0
            if index > 0, let overlap = Self.overlapPrefix(stitched, start: window.start, end: window.prefixEnd) {
                guard overlap.tokens.count < Self.maxTokens - 128 else {
                    throw SplitError.model("SheetSage2's overlap prefix fills the decoder's context at \(Int(window.start)) s")
                }
                (prefix, base) = overlap
            }
            let offset = Int((window.start * Self.sampleRate).rounded())
            var samples = [Float](repeating: 0, count: Self.windowSamples)
            let count = max(0, min(Self.windowSamples, audio.count - offset))
            samples.replaceSubrange(0..<count, with: audio[offset..<offset + count])
            let stop = window.generationStop ?? min(duration - window.start, Self.windowSeconds)
            let tokens = try await generate(samples, prefix: prefix, stop: stop) { done in
                progress((Double(index) + done) / Double(plan.count))
            }
            let events = SheetSageTokens.events(tokens)
            stitched += Self.stitch(events, window: window, duration: duration, base: base)
        }
        stitched.sort { ($0.time, $0.globalSubbeat) < ($1.time, $1.globalSubbeat) }
        return Transcription(events: stitched, duration: duration)
    }

    // MARK: - Decoding

    /// Tokens for one window, continuing `prefix`, until the grammar ends the sequence, a timestamp reaches
    /// `stop` seconds, or the context is full. `progress` gets the fraction of `stop` decoded.
    func generate(
        _ samples: [Float], prefix: [Int], stop: Double, progress: (Double) -> Void
    ) async throws -> [Int] {
        let outputs = try await encoder.run(["samples": MLTensor(shape: [1, Self.windowSamples], scalars: samples)])
        guard let cross = outputs["cross"] else { throw SplitError.model("the SheetSage2 encoder returned no cross output") }
        let values = await cross.shapedArray(of: Float16.self).scalars
        let state = decoder.makeState()
        let slab = Self.heads * Self.frames * Self.headWidth
        for layer in 0..<Self.layers {
            for (offset, name) in ["cross_k_\(layer)", "cross_v_\(layer)"].enumerated() {
                let start = (2 * layer + offset) * slab
                try state.withMultiArray(for: name) { try Self.copy(values[start..<start + slab], into: $0) }
            }
        }

        var grammar = SheetSageTokens.Grammar()
        for token in prefix.drop(while: { $0 != SheetSageTokens.out }).dropFirst() { _ = grammar.update(token) }
        var tokens = prefix
        let token = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let position = try MLMultiArray(shape: [1], dataType: .int32)
        let zeros = UnsafeMutableBufferPointer<Float16>.allocate(capacity: Self.maxTokens)
        zeros.initialize(repeating: 0)
        defer { zeros.deallocate() }

        for step in 0..<Self.maxTokens {
            token[0] = NSNumber(value: tokens[step])
            position[0] = NSNumber(value: step)
            // Zeros whose length, step + 1, tells the model where to write this step's keys and values.
            let mask = try MLMultiArray(
                dataPointer: zeros.baseAddress!, shape: [1, 1, 1, NSNumber(value: step + 1)], dataType: .float16,
                strides: [NSNumber(value: step + 1), NSNumber(value: step + 1), NSNumber(value: step + 1), 1])
            let inputs = try MLDictionaryFeatureProvider(dictionary: ["token": token, "position": position, "mask": mask])
            let output = try decoder.run(inputs, state: state)
            if step + 1 < tokens.count { continue }
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                throw SplitError.model("the SheetSage2 decoder returned no logits")
            }
            let next = Self.argmax(logits, over: grammar.allowed())
            tokens.append(next)
            var finished = grammar.update(next)
            if !finished, SheetSageTokens.time.contains(next) {
                let seconds = Double(next - SheetSageTokens.time.lowerBound) / SheetSageTokens.timeHz
                progress(min(seconds / stop, 1))
                if seconds >= stop { (tokens, finished) = (tokens + [SheetSageTokens.eos], true) }
            }
            if finished { return tokens }
            if tokens.count >= Self.maxTokens { break }
        }
        return tokens + [SheetSageTokens.eos]
    }

    /// The highest of `logits` in `ranges`.
    private static func argmax(_ logits: MLMultiArray, over ranges: [Range<Int>]) -> Int {
        logits.withUnsafeBufferPointer(ofType: Float.self) { values in
            var best = (index: ranges[0].lowerBound, value: -Float.infinity)
            for range in ranges {
                var value: Float = 0
                var index: vDSP_Length = 0
                vDSP_maxvi(values.baseAddress! + range.lowerBound, 1, &value, &index, vDSP_Length(range.count))
                if value > best.value { best = (range.lowerBound + Int(index), value) }
            }
            return best.index
        }
    }

    /// Copies one layer's keys or values, [heads, frames, width] in row-major order, into a state buffer of the
    /// same shape, whatever its strides.
    private static func copy(_ values: ArraySlice<Float16>, into array: MLMultiArray) throws {
        let shape = array.shape.map(\.intValue)
        guard shape == [1, heads, frames, headWidth], array.dataType == .float16 else {
            throw SplitError.model("the SheetSage2 decoder's cross state has shape \(shape)")
        }
        array.withUnsafeMutableBufferPointer(ofType: Float16.self) { destination, strides in
            values.withUnsafeBufferPointer { source in
                for head in 0..<heads {
                    for frame in 0..<frames {
                        let from = (head * frames + frame) * headWidth
                        let to = head * strides[1] + frame * strides[2]
                        for k in 0..<headWidth { destination[to + k * strides[3]] = source[from + k] }
                    }
                }
            }
        }
    }

    // MARK: - Windows

    struct Window: Equatable {
        var start: Double
        var end: Double
        /// Events from here...
        var acceptStart: Double
        /// ...up to here are kept.
        var acceptEnd: Double
        /// Where the prefix carried over from earlier windows ends.
        var prefixEnd: Double
        /// Seconds into the window at which decoding stops; nil for the last window.
        var generationStop: Double?
    }

    /// `sliding_window_plan` with SheetSage2's default overlap (200 s) and lookahead (100 s).
    static func windows(duration: Double, overlap: Double = 200, lookahead: Double = 100) -> [Window] {
        var (start, accepted) = (0.0, 0.0)
        var plan: [Window] = []
        while true {
            let last = start + windowSeconds >= duration - 1e-6
            let acceptEnd = last ? duration : start + windowSeconds - lookahead
            plan.append(Window(
                start: start, end: min(duration, start + windowSeconds), acceptStart: accepted, acceptEnd: acceptEnd,
                prefixEnd: accepted, generationStop: last ? nil : windowSeconds - lookahead))
            if last { return plan }
            accepted = acceptEnd
            start = min(start + windowSeconds - overlap, duration - windowSeconds)
        }
    }

    /// Seconds into the window for a subbeat: piecewise linear through the events' timestamps, extended at the
    /// median seconds per subbeat beyond them (`event_time_map`).
    static func timeMap(_ events: [SheetSageTokens.Event], window: Double = windowSeconds) -> (Double) -> Double {
        var anchors: [Int: Double] = [:]
        for event in events { if let time = event.timestamp { anchors[event.subbeat] = time } }
        let steps = anchors.keys.sorted()
        let times = steps.map { anchors[$0]! }
        guard let first = steps.first, let last = steps.last else {
            return { min(window, max(0, $0 * 0.125)) }
        }
        var perStep = 0.125
        if steps.count >= 2 {
            let rates = (1..<steps.count).map { (times[$0] - times[$0 - 1]) / Double(max(steps[$0] - steps[$0 - 1], 1)) }.sorted()
            let middle = rates.count / 2
            let median = rates.count % 2 == 1 ? rates[middle] : (rates[middle - 1] + rates[middle]) / 2
            if median.isFinite, median > 0 { perStep = median }
        }
        return { step in
            if step <= Double(first) { return min(max(times[0] + (step - Double(first)) * perStep, 0), window) }
            if step >= Double(last) { return min(max(times[times.count - 1] + (step - Double(last)) * perStep, 0), window) }
            // As np.interp: from the last anchor at or before the step, so an anchor's own step gets its time.
            let lower = steps.lastIndex { Double($0) <= step }!
            let (s0, s1) = (Double(steps[lower]), Double(steps[lower + 1]))
            return times[lower] + (times[lower + 1] - times[lower]) / (s1 - s0) * (step - s0)
        }
    }

    /// The window's events between its accept bounds, with times in the song, global subbeats from `base`, and
    /// each melody note's end (`stitched_window_events`).
    static func stitch(
        _ events: [SheetSageTokens.Event], window: Window, duration: Double, base: Int
    ) -> [SheetSageTokens.Event] {
        let time = timeMap(events)
        let epsilon = 1e-4
        return events.compactMap { event in
            let absolute = window.start + time(Double(event.subbeat))
            guard absolute >= window.acceptStart - epsilon, absolute < window.acceptEnd - epsilon,
                  absolute < duration - epsilon else { return nil }
            var event = event
            event.time = min(max(absolute, 0), duration)
            event.globalSubbeat = base + event.subbeat
            event.noteEnds = event.notes.map { note in
                min(duration, max(event.time + 0.04, window.start + time(Double(event.subbeat + note.subbeats))))
            }
            return event
        }
    }

    /// The prompt and the events already kept in [start, end), from the first one with a beat, re-timed to the
    /// window and carrying the section, key, chord and meter in force before them (`build_overlap_prefix_tokens`),
    /// with the global subbeat of the first. Nil when no kept event in the overlap has a beat.
    static func overlapPrefix(
        _ stitched: [SheetSageTokens.Event], start: Double, end: Double
    ) -> (tokens: [Int], base: Int)? {
        let epsilon = 1e-4
        var source = stitched.filter { start - epsilon <= $0.time && $0.time < end - epsilon }
            .sorted { ($0.globalSubbeat, $0.time) < ($1.globalSubbeat, $1.time) }
        guard let first = source.firstIndex(where: { $0.tokens[.timestamp] != nil || $0.tokens[.rhythm] != nil }) else {
            return nil
        }
        source.removeFirst(first)
        let base = source[0].globalSubbeat

        // What was in force at the first event: the latest section, key and chord, and meter, in stitched order.
        var context: [SheetSageTokens.Field: [Int]] = [:]
        var meter: Int?
        for event in stitched where event.time <= source[0].time + 1e-6 {
            for field in [SheetSageTokens.Field.structure, .key, .chord] {
                if let tokens = event.tokens[field] { context[field] = tokens }
            }
            if let token = event.tokens[.rhythm]?.first(where: SheetSageTokens.meter.contains) { meter = token }
        }

        var prefix = source.map { event in
            var event = event
            event.subbeat = event.globalSubbeat - base
            if event.tokens[.timestamp] != nil {
                event.tokens[.timestamp] = [SheetSageTokens.timeToken(seconds: event.time - start)]
            }
            return event
        }
        for (field, tokens) in context where prefix[0].tokens[field] == nil { prefix[0].tokens[field] = tokens }
        if let meter, let rhythm = prefix[0].tokens[.rhythm], !rhythm.contains(where: SheetSageTokens.meter.contains),
           rhythm.contains(where: SheetSageTokens.eighth.contains) {
            prefix[0].tokens[.rhythm] = [meter] + rhythm
        }
        return (SheetSageTokens.encode(prefix), base)
    }
}
