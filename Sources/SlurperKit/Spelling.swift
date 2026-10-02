/// Key and chord names spelled for the local key, and chord pitches, ported from SheetSage2's
/// `chord_spelling_sheetsage2.py` and `midi_sheetsage2.py`. The model predicts roots with sharps; spelling
/// changes only names (D#:maj becomes Eb:maj in C minor), never pitches, qualities or inversions.
enum Spelling {
    /// SheetSage2's canonical key names, by tonic pitch class.
    private static let majorKeys = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    private static let minorKeys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

    static func keyName(tonic: Int, minor: Bool) -> String {
        minor ? "\(minorKeys[tonic]):minor" : "\(majorKeys[tonic]):major"
    }

    /// The pitch class of a note name such as "C", "F#" or "Bbb", or nil.
    static func pitchClass(_ name: Substring) -> Int? {
        guard let letter = name.first.flatMap({ letters.firstIndex(of: $0) }) else { return nil }
        let accidentals = name.dropFirst()
        guard accidentals.allSatisfy({ $0 == "#" || $0 == "b" }) else { return nil }
        let offset = accidentals.count { $0 == "#" } - accidentals.count { $0 == "b" }
        return ((naturals[letter] + offset) % 12 + 12) % 12
    }

    private static let naturals = [0, 2, 4, 5, 7, 9, 11]
    private static let letters = Array("CDEFGAB")
    /// Chord tones as (scale degree - 1, accidental) from the root, for each quality in the vocabulary.
    private static let qualities: [String: [(Int, Int)]] = [
        "maj": [(0, 0), (2, 0), (4, 0)], "min": [(0, 0), (2, -1), (4, 0)], "aug": [(0, 0), (2, 0), (4, 1)],
        "dim": [(0, 0), (2, -1), (4, -1)], "sus4": [(0, 0), (3, 0), (4, 0)],
        "sus4(b7)": [(0, 0), (3, 0), (4, 0), (6, -1)], "sus2": [(0, 0), (1, 0), (4, 0)],
        "7": [(0, 0), (2, 0), (4, 0), (6, -1)], "maj7": [(0, 0), (2, 0), (4, 0), (6, 0)],
        "min7": [(0, 0), (2, -1), (4, 0), (6, -1)], "minmaj7": [(0, 0), (2, -1), (4, 0), (6, 0)],
        "maj6": [(0, 0), (2, 0), (4, 0), (5, 0)], "min6": [(0, 0), (2, -1), (4, 0), (5, 0)],
        "dim7": [(0, 0), (2, -1), (4, -1), (6, -2)], "hdim7": [(0, 0), (2, -1), (4, -1), (6, -1)],
    ]
    /// Each letter's place on the circle of fifths (F = 0 ... B = 6), and each major-scale degree's offset in fifths.
    private static let fifths = [1, 3, 5, 0, 2, 4, 6]
    private static let degreeFifths = [0, 2, 4, -1, 1, 3, 5]

    /// The tones of `quality` on a root spelled as (letter index, accidental), as (letter index, accidental).
    static func tones(root: (letter: Int, accidental: Int), quality: String) -> [(letter: Int, accidental: Int)] {
        let rootPosition = fifths[root.letter] + root.accidental * 7
        return (qualities[quality] ?? []).map { degree, accidental in
            let letter = (root.letter + degree) % 7
            let position = rootPosition + degreeFifths[degree] + accidental * 7
            let steps = position - fifths[letter]
            return (letter, steps >= 0 ? steps / 7 : -((-steps + 6) / 7))
        }
    }

    /// How far a chord's tones sit outside the key's side of the circle of fifths, counting the root twice.
    private static func score(_ tones: [(letter: Int, accidental: Int)], key: (letter: Int, accidental: Int)) -> Int {
        let keyPosition = fifths[key.letter] + key.accidental * 7
        let distances = tones.map { max(abs(fifths[$0.letter] + $0.accidental * 7 - (keyPosition + 2)) - 3, 0) }
        return distances.reduce(0, +) + (distances.first ?? 0)
    }

    /// `label` ("D#:min7/b7") with its root respelled for `key` ("C:minor"): the spelling whose tones sit closest
    /// to the key on the circle of fifths. "N", "X" and labels it cannot read come back unchanged.
    static func chord(_ label: String, inKey key: String) -> String {
        let parts = label.split(separator: ":", maxSplits: 1)
        let keyParts = key.split(separator: ":")
        guard parts.count == 2, keyParts.count == 2, let rootClass = pitchClass(parts[0]),
              let tonic = pitchClass(keyParts[0]) else { return label }
        let quality = String(parts[1].split(separator: "/", maxSplits: 1)[0])
        let inversion = parts[1].contains("/") ? "/" + parts[1].split(separator: "/", maxSplits: 1)[1] : ""
        guard qualities[quality] != nil else { return label }
        // The key signature's tonic: the relative major's tonic as SheetSage2 spells major keys.
        let major = keyParts[1] == "minor" ? (tonic + 3) % 12 : tonic
        let keyName = majorKeys[major]
        let keySpelling = (letters.firstIndex(of: keyName.first!)!, keyName.count == 1 ? 0 : keyName.hasSuffix("b") ? -1 : 1)
        // The first of the best-scoring letters, as numpy's argmin picks.
        let best = (0..<7).map { letter in (letter: letter, accidental: (rootClass - naturals[letter] + 18) % 12 - 6) }
            .min { score(tones(root: $0, quality: quality), key: keySpelling) < score(tones(root: $1, quality: quality), key: keySpelling) }!
        let accidentals = String(repeating: best.accidental < 0 ? "b" : "#", count: abs(best.accidental))
        return "\(letters[best.letter])\(accidentals):\(quality)\(inversion)"
    }

    /// Chord intervals above the root, as mir_eval spells the vocabulary's qualities.
    private static let intervals: [String: [Int]] = [
        "maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8], "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10], "7": [0, 4, 7, 10], "hdim7": [0, 3, 6, 10], "dim7": [0, 3, 6, 9],
        "minmaj7": [0, 3, 7, 11], "sus2": [0, 2, 7], "sus4": [0, 5, 7], "sus4(b7)": [0, 5, 7, 10],
        "maj6": [0, 4, 7, 9], "min6": [0, 3, 7, 9],
    ]
    private static let bassDegrees = ["2": 2, "b3": 3, "3": 4, "5": 7, "b7": 10, "7": 11]

    /// MIDI pitches for a chord: the bass (root or inversion) in the octave from C2, the chord tones and an
    /// inversion's bass from C3. Empty for "N" and labels it cannot read.
    static func pitches(ofChord label: String) -> [Int] {
        let parts = label.split(separator: ":", maxSplits: 1)
        guard parts.count == 2, let root = pitchClass(parts[0]) else { return [] }
        let halves = parts[1].split(separator: "/", maxSplits: 1)
        guard let chord = intervals[String(halves[0])] else { return [] }
        let bass = halves.count == 2 ? bassDegrees[String(halves[1])] ?? 0 : 0
        let upper = Set(chord + [bass]).map { 48 + root + $0 }
        return Array(Set(upper + [36 + (root + bass) % 12])).sorted()
    }
}
