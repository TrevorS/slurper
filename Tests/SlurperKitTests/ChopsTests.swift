import Foundation
import Testing
@testable import SlurperKit

struct ChopsTests {
    @Test func cutFadesBothEndsExceptAtTheStartOfTheAudio() {
        let ones = [[Float](repeating: 1, count: 44_100)]
        let chops = Chops(folder: "kit", pieces: [], fadeIn: 0.003, fadeOut: 0.005)
        let piece = chops.cut(ones, 1_000..<3_000, sampleRate: testRate)[0]
        #expect(piece.count == 2_000)
        #expect(piece[0] == 0 && piece[66] == 0.5 && piece[132] == 1)
        #expect(piece[1_000] == 1 && piece[1_890] < 1 && piece.last == 0)
        #expect(chops.cut(ones, 0..<2_000, sampleRate: testRate)[0][0] == 1)
    }

    @Test func scalesPiecesToAnotherSampleRate() {
        let chops = Chops(folder: "kit", pieces: [.init(name: "01", range: 4_410..<44_100)], fadeIn: 0, fadeOut: 0)
        #expect(chops.scaled(by: 48_000 / 44_100, frames: 47_999).pieces == [.init(name: "01", range: 4_800..<47_999)])
    }

    @Test func labelsPadToTheWidestNumber() {
        #expect(Chops.label(7, of: 12, digits: 3) == "007")
        #expect(Chops.label(7, of: 1_200, digits: 3) == "0007")
    }
}
