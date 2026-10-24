import Foundation
import Testing

@testable import PipecatServiceToolbar
@testable import PipecatToolbarCore

/// `ServiceModel`'s lifecycle edges that race a separate task: a server's exit
/// (reported after its stdout drains) and app quit during an install.
@MainActor @Suite struct ServiceModelTests {
    static let failure = ServiceModel.Failure(kind: .install, title: "Couldn't remove", message: "no")
    static let phases: [ServiceModel.Phase] = [
        .checking, .unavailable("old"), .notInstalled, .needsModel, .installing(.model), .installed,
        .starting, .running, .stopping, .removing, .failed(failure),
    ]

    /// {phase} x {requested exit, crash} x {the exiting server is current, or
    /// already replaced}. A requested exit only completes a Stop; in particular
    /// one reported after a removal finished (`.notInstalled`) or failed keeps
    /// that phase. A crash of the current server fails it; a replaced server's
    /// exit changes nothing.
    @Test func everyExitAgainstEveryPhase() {
        for phase in Self.phases {
            for requested in [true, false] {
                for current in [true, false] {
                    let model = ServiceModel(info: Service.tone.info)
                    model.setPreviewState(phase: phase)
                    let sub = Subprocess(.init(executable: URL(filePath: "/bin/true"), arguments: [], environment: [:]))
                    model.server = current ? sub : Subprocess(sub.spec)
                    model.stopRequested = requested
                    model.serverExited(sub, status: 0)
                    let label = "\(phase) requested=\(requested) current=\(current)"
                    switch (current, requested) {
                    case (false, _): #expect(model.phase == phase, "\(label)")
                    case (true, true): #expect(model.phase == (phase == .stopping ? .installed : phase), "\(label)")
                    case (true, false):
                        guard case .failed(let f) = model.phase, case .crashed(status: 0) = f.kind else {
                            Issue.record("\(label): \(model.phase)")
                            continue
                        }
                    }
                    #expect(current ? model.server == nil : model.server != nil, "\(label)")
                }
            }
        }
    }

    /// The reported order: remove() -> shutdown() returns -> removal and
    /// refresh finish (`.notInstalled`) -> the drained exit arrives.
    @Test func aLateExitDoesNotReinstallARemovedService() {
        let model = ServiceModel(info: Service.tone.info)
        let sub = Subprocess(.init(executable: URL(filePath: "/bin/true"), arguments: [], environment: [:]))
        model.setPreviewState(phase: .removing)
        model.server = sub
        model.stopRequested = true
        model.setPreviewState(phase: .notInstalled)  // refresh() after manager.remove
        model.serverExited(sub, status: 15)
        #expect(model.phase == .notInstalled)
    }

    /// App quit cancels an Apple Speech model install and waits for it to unwind.
    @Test func shutdownCancelsAndAwaitsAnAppleSpeechInstall() async {
        let model = ServiceModel(info: Service.appleSpeech.info)
        var ended = false
        model.appleSpeechInstall = { _ in
            defer { ended = true }
            try await Task.sleep(for: .seconds(30))
        }
        model.install()
        await Task.yield()
        let clock = ContinuousClock()
        let start = clock.now
        await model.shutdown()
        #expect(ended, "shutdown returned with the install still running")
        #expect(clock.now - start < .seconds(2))
    }

    /// An install that ignores cancellation holds quit for at most `installDrain`.
    @Test func shutdownWaitsForAStuckInstallOnlySoLong() async {
        let model = ServiceModel(info: Service.appleSpeech.info)
        model.installDrain = .milliseconds(200)
        model.appleSpeechInstall = { _ in
            let deadline = ContinuousClock.now + .seconds(3)
            while ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(20)) }
        }
        model.install()
        let clock = ContinuousClock()
        let start = clock.now
        await model.shutdown()
        #expect(clock.now - start < .seconds(2))
    }
}
