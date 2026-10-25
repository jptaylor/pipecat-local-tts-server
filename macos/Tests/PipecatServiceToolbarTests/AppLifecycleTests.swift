import Testing

@testable import PipecatToolbarCore

/// `AppLifecycle.reduce`: the app-wide bootstrap / Reset / quit machine
/// (`AppModel` is its adapter).
@Suite struct AppLifecycleTests {
    static let phases: [AppPhase] = [.idle, .bootstrapping, .resetting, .ready, .failed("x"), .shuttingDown]
    static let events: [AppEvent] = [
        .bootstrap, .retry, .reset(servicesBusy: false), .reset(servicesBusy: true), .quit, .bootstrapDone,
        .bootstrapFailed("e"), .resetDone,
    ]

    /// The accepted transitions; every other (phase, event) pair is refused:
    /// no phase change, no effect.
    static func expected(_ phase: AppPhase, _ event: AppEvent) -> (AppPhase, [AppEffect])? {
        switch (phase, event) {
        case (.shuttingDown, _): nil
        case (_, .quit): (.shuttingDown, [.shutdown])
        case (.idle, .bootstrap): (.bootstrapping, [.runBootstrap])
        case (.failed, .retry): (.bootstrapping, [.runBootstrap])
        case (.ready, .reset(false)), (.failed, .reset(false)): (.resetting, [.runReset])
        case (.bootstrapping, .bootstrapDone): (.ready, [])
        case (.bootstrapping, .bootstrapFailed(let m)): (.failed(m), [])
        case (.resetting, .resetDone): (.bootstrapping, [.runBootstrap])
        default: nil
        }
    }

    /// {every phase} x {every event}: the table above, and the one rule: a job
    /// (bootstrap or Reset) starts only when none runs, and only through a
    /// phase that names it; quit runs the shutdown from every phase but
    /// `shuttingDown`, which nothing leaves.
    @Test func wholeSpace() {
        #expect(AppEvent.Kind.allCases.count == 7)
        #expect(Set(Self.events.map(\.kind)).count == AppEvent.Kind.allCases.count)
        for phase in Self.phases {
            for event in Self.events {
                var p = phase
                let effects = AppLifecycle.reduce(&p, event)
                let label = "\(phase) + \(event)"
                if let (next, work) = Self.expected(phase, event) {
                    #expect(p == next && effects == work, "\(label): \(p) \(effects)")
                } else {
                    #expect(p == phase && effects.isEmpty, "refused: \(label): \(p) \(effects)")
                }
                if effects.contains(.runBootstrap) || effects.contains(.runReset) {
                    #expect(!phase.hasJob || event == .resetDone, "a second job started: \(label)")
                    #expect(p.hasJob, "a job started without a phase naming it: \(label)")
                }
                if phase == .shuttingDown { #expect(p == .shuttingDown && effects.isEmpty, "\(label)") }
            }
        }
    }

    /// The derived flags come from the phase alone.
    @Test func derivedFlags() {
        for phase in Self.phases {
            let failed = if case .failed = phase { true } else { false }
            #expect(phase.canRetryBootstrap == failed, "\(phase)")
            #expect(phase.canReset == (phase == .ready || failed), "\(phase)")
            #expect(phase.hasJob == (phase == .bootstrapping || phase == .resetting), "\(phase)")
            #expect(phase.isChecking == (phase == .idle || phase.hasJob), "\(phase)")
            #expect(phase.setupError == (failed ? "x" : nil), "\(phase)")
        }
    }

    /// Runs `events` from `phase`, returning the phase and every effect issued.
    static func run(_ phase: AppPhase, _ events: [AppEvent]) -> (AppPhase, [AppEffect]) {
        var p = phase
        var all: [AppEffect] = []
        for event in events { all += AppLifecycle.reduce(&p, event) }
        return (p, all)
    }

    @Test func tryAgainTwiceStartsOneBootstrap() {
        let (p, effects) = Self.run(.failed("no uv"), [.retry, .retry])
        #expect(p == .bootstrapping && effects == [.runBootstrap])
    }

    @Test func resetTwiceStartsOneReset() {
        let (p, effects) = Self.run(.ready, [.reset(servicesBusy: false), .reset(servicesBusy: false)])
        #expect(p == .resetting && effects == [.runReset])
        // Its bootstrap is the same job's tail: a Reset during it is refused too.
        let (after, tail) = Self.run(p, [.resetDone, .reset(servicesBusy: false), .retry])
        #expect(after == .bootstrapping && tail == [.runBootstrap])
    }

    /// Quit during Reset's shutdown-and-delete window runs the shutdown (which
    /// waits for the Reset job), and the Reset's end starts no bootstrap after it.
    @Test func quitDuringResetWaitsAndStartsNothing() {
        let (p, effects) = Self.run(.ready, [.reset(servicesBusy: false), .quit, .resetDone])
        #expect(p == .shuttingDown && effects == [.runReset, .shutdown])
    }

    @Test func quitDuringBootstrap() {
        let (p, effects) = Self.run(.idle, [.bootstrap, .quit, .bootstrapDone, .quit, .bootstrapFailed("killed")])
        #expect(p == .shuttingDown && effects == [.runBootstrap, .shutdown])
    }

    /// A Try Again or a second launch while a bootstrap runs is refused, so the
    /// first one's end is the only one (it can't clear a flag another still needs).
    @Test func bootstrapFinishWhileAnotherRequested() {
        let (p, effects) = Self.run(.idle, [.bootstrap, .retry, .bootstrap, .bootstrapFailed("e"), .bootstrapDone])
        #expect(p == .failed("e") && effects == [.runBootstrap])
        let (again, more) = Self.run(p, [.retry, .retry, .bootstrapDone])
        #expect(again == .ready && more == [.runBootstrap])
    }

    /// Quit with nothing running still runs the shutdown: a cache probe (or the
    /// manager's other subprocesses) can be live while no service blocks quit.
    @Test func quitFromReadyRunsTheShutdown() {
        let (p, effects) = Self.run(.ready, [.quit])
        #expect(p == .shuttingDown && effects == [.shutdown])
    }

    /// Reset is refused while a service holds live work.
    @Test func resetRefusedUnderLiveWork() {
        let (p, effects) = Self.run(.ready, [.reset(servicesBusy: true)])
        #expect(p == .ready && effects.isEmpty)
    }
}
