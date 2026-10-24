import Foundation
import Testing

@testable import PipecatToolbarCore

/// `ServiceLifecycle.reduce`, the pure state machine behind every service:
/// late ready/exit/timeout events, Stop, Delete and quit against every phase.
@Suite struct ServiceLifecycleTests {
    static let failure = ServiceFailure(kind: .install, title: "Couldn't remove", message: "no")
    static let phases: [ServicePhase] = [
        .checking, .unavailable("old"), .notInstalled, .needsModel, .installing(.model), .installed,
        .starting, .running, .stopping, .removing, .failed(failure),
    ]
    static let current = ServerID(1)
    static let stale = ServerID(0)
    static let date = Date(timeIntervalSince1970: 0)
    static let marker = EnvironmentMarker(service: "tone", fingerprint: "w", python: "3.12", createdAt: date)

    static func state(
        _ phase: ServicePhase, info: ServiceInfo = Service.tone.info, server: ServerID? = current,
        stopRequested: Bool = false, shuttingDown: Bool = false, installing: Bool = false
    ) -> LifecycleState {
        var s = LifecycleState(info: info)
        s.phase = phase
        s.environment = .ready(marker)
        s.requested = true
        s.server = server
        s.stopRequested = stopRequested
        s.shuttingDown = shuttingDown
        s.installing = installing
        return s
    }

    // MARK: - Ported from the app-target ServiceModelTests

    /// {phase} x {requested exit, crash} x {the exiting server is current, or
    /// already replaced}. A requested exit only completes a Stop; in particular
    /// one reported after a removal finished (`.notInstalled`) or failed keeps
    /// that phase. A crash of the current server fails it; a replaced server's
    /// exit changes nothing.
    @Test func everyExitAgainstEveryPhase() {
        for phase in Self.phases {
            for requested in [true, false] {
                for current in [true, false] {
                    var s = Self.state(phase, server: current ? Self.current : ServerID(2), stopRequested: requested)
                    _ = ServiceLifecycle.reduce(&s, .exited(Self.current, status: 0, logTail: []))
                    let label = "\(phase) requested=\(requested) current=\(current)"
                    switch (current, requested) {
                    case (false, _): #expect(s.phase == phase, "\(label)")
                    case (true, true): #expect(s.phase == (phase == .stopping ? .installed : phase), "\(label)")
                    case (true, false):
                        guard case .failed(let f) = s.phase, case .crashed(status: 0) = f.kind else {
                            Issue.record("\(label): \(s.phase)")
                            continue
                        }
                    }
                    #expect(current ? s.server == nil : s.server != nil, "\(label)")
                }
            }
        }
    }

    /// The reported order: remove() -> shutdown() returns -> removal and
    /// refresh finish (`.notInstalled`) -> the drained exit arrives.
    @Test func aLateExitDoesNotReinstallARemovedService() {
        var s = Self.state(.installed)
        _ = ServiceLifecycle.reduce(&s, .removeRequested(deleteModel: true, managerAvailable: true))
        _ = ServiceLifecycle.reduce(&s, .removalResumed(deleteModel: true))
        #expect(ServiceLifecycle.reduce(&s, .shutdownServers) == [.detachTryIt, .stopServer(Self.current, wait: true)])
        #expect(ServiceLifecycle.reduce(&s, .removalFinished) == [.clearLog, .refresh, .familyChanged])
        _ = ServiceLifecycle.reduce(&s, .refreshed(.notInstalled, environment: .missing, requested: false))
        _ = ServiceLifecycle.reduce(&s, .exited(Self.current, status: 15, logTail: []))
        #expect(s.phase == .notInstalled)
    }

    // MARK: - The whole state space

    /// Every event, for every phase, with the server current or late (a stale
    /// id), and with or without a Stop/removal (`stopRequested`) and app quit
    /// (`shuttingDown`) under way. Each row names the property that must hold.
    @Test func everyEventAgainstEveryState() {
        let tail = ["last line"]
        for phase in Self.phases {
            for late in [false, true] {
                for stopping in [false, true] {
                    for quitting in [false, true] {
                        let id = late ? Self.stale : Self.current
                        let label = "\(phase) late=\(late) stop=\(stopping) quit=\(quitting)"
                        func run(_ event: LifecycleEvent, info: ServiceInfo = Service.tone.info, installing: Bool = false)
                            -> (before: LifecycleState, after: LifecycleState, effects: [LifecycleEffect])
                        {
                            let before = Self.state(
                                phase, info: info, stopRequested: stopping, shuttingDown: quitting, installing: installing)
                            var after = before
                            let effects = ServiceLifecycle.reduce(&after, event)
                            return (before, after, effects)
                        }

                        // A ready line flips only the current server's `.starting` to running.
                        let ready = run(.ready(id, port: 8665, pid: 9, rate: 24000, model: nil, at: Self.date))
                        if !late && phase == .starting {
                            #expect(ready.after.phase == .running && ready.effects == [.openTryIt(port: 8665)], "\(label)")
                        } else {
                            #expect(ready.after == ready.before && ready.effects.isEmpty, "ready \(label)")
                        }

                        // The ready timeout stops only the current server still starting.
                        let timeout = run(.readyTimedOut(id))
                        if !late && phase == .starting {
                            #expect(timeout.effects == [.stopServer(Self.current, wait: false)], "\(label)")
                        } else {
                            #expect(timeout.after == timeout.before && timeout.effects.isEmpty, "timeout \(label)")
                        }

                        // An exit: a late one changes nothing; a requested one only completes a Stop.
                        let exit = run(.exited(id, status: 1, logTail: tail))
                        if late {
                            #expect(exit.after == exit.before && exit.effects.isEmpty, "exit \(label)")
                        } else if stopping {
                            #expect(exit.after.phase == (phase == .stopping ? .installed : phase), "exit \(label)")
                        } else {
                            #expect(exit.after.phase == .failed(ServiceFailure(
                                kind: .crashed(status: 1), title: "Server stopped unexpectedly",
                                message: "The Tone server exited with status 1.", logTail: tail)), "exit \(label)")
                        }
                        if !late { #expect(exit.after.server == nil && exit.after.running == nil, "\(label)") }

                        // Start never runs during a removal or with a server up.
                        let start = run(.startRequested(appAvailable: true, managerAvailable: true, portFree: true))
                        #expect(start.after == start.before && start.effects.isEmpty, "start with a server \(label)")
                        var idle = Self.state(phase, server: nil)
                        let startIdle = ServiceLifecycle.reduce(
                            &idle, .startRequested(appAvailable: true, managerAvailable: true, portFree: true))
                        #expect(startIdle == (phase == .removing ? [] : [.launch(port: 8665)]), "start \(label)")

                        // Stop moves the current server to `.stopping` from any phase.
                        let stop = run(.stopRequested)
                        #expect(stop.after.phase == .stopping && stop.after.stopRequested, "stop \(label)")

                        // An install that ends while a removal owns the phase leaves it alone.
                        for outcome in [InstallOutcome.cancelled, .failed(stage: .model, message: "x", logTail: [])] {
                            let ended = run(.installEnded(outcome, then: .refresh), installing: true)
                            #expect(!ended.after.installing, "\(label)")
                            if phase == .removing {
                                #expect(ended.after.phase == .removing && ended.effects.isEmpty, "install \(label)")
                            }
                        }

                        // Quit cancelled the Apple Speech install: nothing to show.
                        for outcome in [InstallOutcome.cancelled, .failed(stage: .model, message: "x", logTail: [])] {
                            let ended = run(.appleSpeechInstallEnded(outcome), info: Service.appleSpeech.info, installing: true)
                            if quitting {
                                #expect(ended.after.phase == phase && ended.effects.isEmpty, "apple \(label)")
                            } else if outcome == .cancelled {
                                #expect(ended.after.phase == phase && ended.effects == [.refresh], "apple \(label)")
                            }
                        }

                        // A second Delete during a removal is a no-op; otherwise Delete takes the phase.
                        let remove = run(.removeRequested(deleteModel: true, managerAvailable: true))
                        if phase == .removing {
                            #expect(remove.after == remove.before && remove.effects.isEmpty, "remove \(label)")
                        } else {
                            #expect(remove.after.phase == .removing, "remove \(label)")
                            #expect(remove.effects.first == .cancelInstall, "remove \(label)")
                        }

                        // A sibling's change re-reads only settled phases.
                        let sibling = run(.environmentChanged(.missing))
                        let settled = [.checking, .notInstalled, .needsModel, .installed].contains(phase)
                        #expect(sibling.effects == (settled ? [.refresh] : []), "sibling \(label)")
                        #expect(sibling.after.environment == (settled ? sibling.before.environment : .missing), "\(label)")
                    }
                }
            }
        }
    }

    /// Quit cancels only an Apple Speech install, and waits for it boundedly.
    @Test func shutdownCancelsOnlyAnAppleSpeechInstall() {
        var apple = Self.state(.installing(.model), info: Service.appleSpeech.info, server: nil, installing: true)
        #expect(ServiceLifecycle.reduce(&apple, .shutdownRequested) == [.cancelInstall, .awaitInstall(bounded: true)])
        #expect(apple.shuttingDown)
        var python = Self.state(.installing(.model), server: nil, installing: true)
        #expect(ServiceLifecycle.reduce(&python, .shutdownRequested).isEmpty)
        #expect(!python.shuttingDown)
    }

    /// The native server: a bind that lands after Stop is discarded.
    @Test func aLateNativeBindIsDiscarded() {
        var s = Self.state(.installed, info: Service.appleSpeech.info, server: nil)
        _ = ServiceLifecycle.reduce(&s, .nativeLaunched(Self.current))
        #expect(s.phase == .starting)
        #expect(ServiceLifecycle.reduce(&s, .stopRequested) == [.stopNative, .tearDown(note: "stopped")])
        #expect(s.phase == .installed)
        #expect(ServiceLifecycle.reduce(&s, .nativeBound(Self.current, port: 9465, pid: 1, at: Self.date)) == [.discardNative(Self.current)])
        #expect(s.phase == .installed)
    }

    /// A supervised error decides how the exit reads.
    @Test func exitAfterSupervisedErrors() {
        var busy = Self.state(.starting)
        _ = ServiceLifecycle.reduce(&busy, .supervisedError(Self.current, SupervisedError(code: "EADDRINUSE", message: nil)))
        #expect(ServiceLifecycle.reduce(&busy, .exited(Self.current, status: 1, logTail: [])).last == .probePortOwner(port: 8665))
        var missing = Self.state(.starting)
        _ = ServiceLifecycle.reduce(&missing, .supervisedError(Self.current, SupervisedError(code: "ModuleNotFoundError", message: "m")))
        _ = ServiceLifecycle.reduce(&missing, .exited(Self.current, status: 1, logTail: []))
        #expect(missing.phase == .failed(ServiceFailure(kind: .startup(code: "ModuleNotFoundError"), title: "Environment incomplete", message: "m")))
    }

    /// The install's mode and first stage, and an outdated environment refreshed before start.
    @Test func installModesAndOutdatedStart() {
        #expect(ServiceLifecycle.installMode(environment: .missing, phase: .needsModel) == .full)
        #expect(ServiceLifecycle.installMode(environment: .ready(Self.marker), phase: .needsModel) == .model)
        let download = ServiceFailure(kind: .download, title: "", message: "")
        #expect(ServiceLifecycle.installMode(environment: .ready(Self.marker), phase: .failed(download)) == .model)
        #expect(ServiceLifecycle.firstStage(mode: .full, environment: .missing) == .python)
        var outdated = Self.state(.installed, server: nil)
        outdated.environment = .outdated(Self.marker)
        #expect(
            ServiceLifecycle.reduce(&outdated, .startRequested(appAvailable: true, managerAvailable: true, portFree: true))
                == [.runInstall(.packages, then: .refreshThenStart)])
        #expect(outdated.phase == .installing(.packages) && outdated.installing)
        #expect(ServiceLifecycle.reduce(&outdated, .installEnded(.finished, then: .refreshThenStart)) == [.refresh, .startIfNotCancelled, .familyChanged])
    }
}
