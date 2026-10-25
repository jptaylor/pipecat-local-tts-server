import Foundation
import Testing

@testable import PipecatToolbarCore

@Suite struct STTSampleClipTests {
    @Test func clipIsAWholeNumberOfSixteenKilohertzMonoSamplesAndLoud() {
        let pcm = STTSampleClip.pcm16
        #expect(pcm.count % 2 == 0)
        let seconds = Double(pcm.count) / 2 / 16_000
        #expect((2.0...4.0).contains(seconds), "clip is \(seconds) s")
        let peak = pcm.withUnsafeBytes { raw in
            raw.bindMemory(to: Int16.self).map { abs(Int($0)) }.max() ?? 0
        }
        #expect(peak > 5_000, "the clip is silent (peak \(peak))")
    }

    @Test(arguments: [
        "Pipecat runs speech services right here on this Mac.",
        "pipecat runs speech services right here on this mac",
        "Pipe cat runs speech services, right here on this Mac!",
        "  Pipecat runs speech services right here on this Max.  ",  // one slip
    ])
    func acceptsTheSentenceAndSmallSlips(_ transcript: String) {
        #expect(STTSampleClip.matches(transcript))
    }

    @Test(arguments: [
        "",
        "(no speech detected)",
        "Hello from Pipecat.",
        "Pipecat runs speech",  // truncated
        "Something else entirely about the weather today.",
    ])
    func rejectsOtherText(_ transcript: String) {
        #expect(!STTSampleClip.matches(transcript))
    }
}
